"""Check schedule on CPU, and numerical behavior when CUDA is available."""
import torch
import torch.nn.functional as F
from model import ModelNew


def test_schedule():
    for rows in (8,16):
        for row in range(rows):
            taps=[(ir-row,kc) for ir in range(rows+6) for kc in range(7) if row<=ir<row+7]
            assert taps==[(kr,kc) for kr in range(7) for kc in range(7)]
        for tid in range(128):
            base=5+tid*4
            offsets=[0]+[1+p*2+j for p in range(4) for j in range(2)]+[9]
            assert offsets==list(range(10))
            assert all((base+1+p*2)*2%4==0 for p in range(4))
            assert base+9 < 528
        cells=[]
        for sy in range(rows+6):
            cells.extend((sy,3+g*8+j) for g in range(64) for j in range(8))
            cells.extend((sy,k if k<3 else 512+k) for k in range(6))
        assert len(cells)==len(set(cells))==(rows+6)*518
    assert set(ModelNew().state_dict())=={'conv1.weight','conv1.bias'}
    print('CPU checks passed: both row tiles, half2 alignment, union coverage, interface')


def test_cuda():
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cases = [torch.randn(2, 1, 1, 1), torch.randn(1, 7, 9),
             torch.randn(2, 1, 17, 257), torch.randn(1, 1, 48, 769),
             torch.randn(1, 1, 48, 1536), torch.randn(1, 1, 35, 10240),
             torch.randn(2, 1, 17, 516), torch.randn(1, 1, 2, 4),
             torch.randn(2, 1, 35, 518)[:, :, ::2, ::2],
             torch.randn(2, 1, 259, 19).transpose(-1, -2),
             torch.randn(4, 1, 9, 13)[::2], torch.empty(0, 1, 9, 13)]
    count = 0
    for dtype, tolerance in [(torch.float16, 2e-3), (torch.bfloat16, 2e-2),
                             (torch.float32, 2e-5), (torch.float64, 1e-10)]:
        for use_bias in (True, False):
            model = ModelNew().cuda().to(dtype)
            if not use_bias:
                model.conv1.bias = None
            for cpu_x in cases:
                # Explicit strides preserve sliced layouts during the transfer.
                x = torch.empty_strided(cpu_x.shape, cpu_x.stride(), device='cuda', dtype=dtype)
                x.copy_(cpu_x)
                acc_dtype = torch.float64 if dtype == torch.float64 else torch.float32
                with torch.no_grad():
                    actual = model(x)
                    bias = model.conv1.bias
                    expected = F.conv2d(x.to(acc_dtype), model.conv1.weight.to(acc_dtype),
                                        None if bias is None else bias.to(acc_dtype), padding=3).to(dtype)
                torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
                assert actual.shape == x.shape
                count += 1
    print(f'CUDA correctness passed: {count} cases')


if __name__ == '__main__':
    test_schedule()
    if torch.cuda.is_available():
        test_cuda()
    else:
        print('SKIP CUDA correctness: no accessible CUDA device')
