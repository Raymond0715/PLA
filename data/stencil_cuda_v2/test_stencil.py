"""CPU fragment/index checks and CUDA correctness checks, without benchmarking JIT."""
import argparse
import torch
import torch.nn.functional as F
from model import ModelNew


def check_fragments():
    # Reconstruct the exact m16n8k16 register ownership used by the kernel.
    torch.manual_seed(7)
    tile = torch.zeros(14, 272)
    tile[:, 5:267] = torch.randn(14, 262)
    raw = torch.randn(7, 7)
    weight = torch.zeros(8, 80)
    for m in range(8):
        weight[m].view(8, 10)[m // 4:m // 4 + 7, m % 4:m % 4 + 7] = raw
    for warp in range(4):
        for pair in range(4):
            out = torch.zeros(16, 8)
            for kb in range(0, 80, 16):
                a = torch.empty(16, 16)
                b = torch.empty(16, 8)
                for lane in range(32):
                    g, t = lane // 4, lane % 4
                    for r, k in [(g, 2*t), (g+8, 2*t), (g, 2*t+8), (g+8, 2*t+8)]:
                        for e in range(2):
                            kk = kb+k+e
                            a[r, k+e] = tile[2*pair+kk//10, 5+4*(warp*16+r)+kk%10]
                    for k in [2*t, 2*t+8]:
                        b[k:k+2, g] = weight[g, kb+k:kb+k+2]
                out += a @ b
            expected = F.conv2d(tile[None,None,:,5:267], raw[None,None])[0,0]
            for r in range(16):
                for m in range(8):
                    torch.testing.assert_close(out[r,m], expected[2*pair+m//4,4*(warp*16+r)+m%4],atol=2e-5,rtol=2e-5)
    # Coverage includes borders, horizontal CTA tails, and two batches.
    for h,w in [(1,1),(3,5),(4,256),(5,257),(7,259),(8,256),(9,257),(15,519)]:
        count = torch.zeros((2,h,w),dtype=torch.int32)
        for n in range(2):
            for by in range(0,h,8):
                for bx in range(0,w,256):
                    for warp in range(4):
                        for lane in range(32):
                            g,t = lane//4,lane%4
                            for pair in range(4):
                                row=by+2*pair+t//2
                                for rr in (g,g+8):
                                    col=bx+4*(warp*16+rr)+2*(t%2)
                                    for cc in (col,col+1):
                                        if row<h and cc<w: count[n,row,cc]+=1
        assert torch.all(count==1)
    # Validate async middle vectors and scalar halos cover each logical
    # cell exactly once, and every async shared destination is aligned.
    counts = torch.zeros(14, 272, dtype=torch.int32)
    for task in range(14 * 32):
        sy, vector = divmod(task, 32)
        physical = 3 + 8 * vector + 5
        assert (2 * (sy * 272 + physical)) % 16 == 0
        counts[sy, physical:physical+8] += 1
    for task in range(14 * 6):
        sy, item = divmod(task, 6)
        sx = item if item < 3 else 256 + item
        counts[sy, sx+5] += 1
    assert torch.all(counts[:, 5:267] == 1)
    assert counts.sum() == 3668
    print('PASS CPU MMA fragment mapping and output coverage')


def check_cuda():
    if not torch.cuda.is_available():
        print('SKIP CUDA execution: no available GPU')
        return
    torch.backends.cudnn.allow_tf32 = False
    for dtype in (torch.float16,torch.float32):
        model=ModelNew().cuda().to(dtype).eval()
        for h,w in [(1,1),(2,3),(3,5),(4,256),(5,257),(7,259),(8,256),(9,257),(15,519),(32,1024)]:
            # Exercise non-contiguous inputs as well as full/boundary tiles.
            x=torch.randn(2,1,h,w*2,device='cuda',dtype=dtype)[:,:,:,::2]
            with torch.no_grad():
                expected=F.conv2d(x.float(),model.conv1_weight.float(),padding=3).to(dtype)+model.conv1_bias
                actual=model(x)
            tol=2e-3 if dtype==torch.float16 else 2e-5
            torch.testing.assert_close(actual,expected,atol=tol,rtol=tol)
        with torch.no_grad():
            # Packing must always reflect current parameters.
            model.conv1_weight.zero_()
            model.conv1_bias.fill_(0.25)
            x=torch.randn(1,1,7,11,device='cuda',dtype=dtype)
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream): y=model(x)
            torch.cuda.current_stream().wait_stream(stream)
            torch.testing.assert_close(y,torch.full_like(y,0.25),atol=0,rtol=0)
            for shape in [(0,1,3,4),(1,1,0,4),(1,1,3,0)]:
                assert model(torch.empty(shape,device='cuda',dtype=dtype)).shape==shape
        print('PASS CUDA',dtype,'boundaries/strides/updated weights/stream/empty inputs')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cpu-only',action='store_true')
    args=parser.parse_args()
    check_fragments()
    if not args.cpu_only: check_cuda()
