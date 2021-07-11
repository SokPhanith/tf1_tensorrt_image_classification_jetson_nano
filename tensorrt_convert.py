import argparse
import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
def get_parser():
    parser = argparse.ArgumentParser(description="Build TensorRT Engine")
    parser.add_argument("--model",default="model.uff",help="Path to uff file default[model.uff].")
    parser.add_argument("--output",default="model.engine",help="Path to output engine default[model.engine].")
    parser.add_argument("--input_name",default="input",help="Give name input of Graph default[input].")
    parser.add_argument("--output_name",default="MobilenetV1/Predictions/Reshape_1",help="Give name output of Graph default[MobilenetV1/Predictions/Reshape_1].")
    parser.add_argument("--fp16", action="store_true",help="set --fp16 for Build TensorRT FP16")
    parser.add_argument("--height",type=int,default=224,help="height input image default [224].")
    parser.add_argument("--width",type=int,default=224,help="width input image default [224].")
    parser.add_argument("--batch_size",type=int,default=1,help="batch size input image default [1].")
    parser.add_argument("--channel",type=int,default=3,help="channel input image default [3].")
    return parser
args = get_parser().parse_args()
print(args)
with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
    builder.max_workspace_size = 1<<28
    builder.max_batch_size = args.batch_size
    builder.fp16_mode = args.fp16
    parser.register_input(args.input_name,(args.channel,args.width,args.height))
    parser.register_output(args.output_name)
    parser.parse(args.model,network)
    engine = builder.build_cuda_engine(network)
    buf = engine.serialize()
    with open(args.output, 'wb') as f:
        f.write(buf)

