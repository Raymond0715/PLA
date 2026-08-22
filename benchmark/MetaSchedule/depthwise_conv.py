"""TVM benchmark for NCHW depthwise convolution (channel multiplier one)."""
import json
import numpy as np
from common import parser, run

def main():
    p=parser(__doc__); p.add_argument("--batch",type=int,default=1); p.add_argument("--channels",type=int,default=32); p.add_argument("--height",type=int,default=128); p.add_argument("--width",type=int,default=128); p.add_argument("--kernel",type=int,choices=(3,7),default=3)
    args=p.parse_args()
    n,c,h,w,k,dtype=args.batch,args.channels,args.height,args.width,args.kernel,args.dtype; pad=k//2
    if args.implementation == "dlight" and dtype == "float16" and k == 7:
        print(json.dumps({
            "operator": f"depthwise_{n}_{c}_{h}_{w}_k{k}",
            "implementation": args.implementation,
            "dtype": dtype,
            "status": "skipped",
            "reason": (
                "known TVM 0.26 tirx/DLight lowering bug: CUDA depthwise float16 k7 "
                "mixes int64 and int32 index expressions"
            ),
        }, indent=2))
        return
    from tvm import te, topi
    X=te.placeholder((n,c,h,w),dtype,"X"); W=te.placeholder((c,1,k,k),dtype,"W")
    # Use TOPI's standard depthwise-convolution definition for every dtype.
    # In particular, this gives float16 the same explicit padding stage as
    # float32, which is easier for MetaSchedule to analyze than an inlined
    # boundary check followed by a separate output cast.
    Y=topi.nn.depthwise_conv2d_nchw(X,W,1,pad,1,dtype)
    mod=te.create_prim_func([X,W,Y]).with_attr("global_symbol","main")
    def vendor():
        from tvm.contrib import cudnn
        if not cudnn.exists(): raise SystemExit("vendor requires TVM built with USE_CUDNN=ON and a visible CUDA GPU")
        out=cudnn.conv_forward(X,W,pad,(1,1),(1,1),1,0,-1,dtype,groups=c,verbose=False)
        return te.create_prim_func([X,W,out]).with_attr("global_symbol","main")
    def ref(x,weight):
        xp=np.pad(x.astype("float32"),((0,0),(0,0),(pad,pad),(pad,pad))); out=np.zeros(x.shape,dtype="float32")
        weight=weight.astype("float32")
        for ry in range(k):
            for rx in range(k): out += xp[:,:,ry:ry+h,rx:rx+w]*weight[:,0,ry,rx][None,:,None,None]
        return out.astype(x.dtype)
    run(name=f"depthwise_{n}_{c}_{h}_{w}_k{k}",mod=mod,shapes=[(n,c,h,w),(c,1,k,k),(n,c,h,w)],reference=ref,args=args,vendor_factory=vendor)

if __name__ == "__main__": main()
