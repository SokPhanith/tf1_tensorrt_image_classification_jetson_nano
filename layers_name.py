import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = '1'
import tensorflow as tf
import sys
if len(sys.argv) == 2:
    model_path = sys.argv[1]
    p = 'none'
elif len(sys.argv) == 3:
    model_path = sys.argv[1]
    p = sys.argv[2]
else:
    print('run this python like : ')
    print('python3 layers_name.py <<frozen_graph.pb model path or inf_graph.pb path>>')
    print('or python3 layers_name.py <<frozen_graph.pb model path or inf_graph.pb path>> all')
    sys.exit()
gf = tf.compat.v1.GraphDef()
gf.ParseFromString(open(model_path,'rb').read())
node_name = [n.name for n in gf.node if n.op]
if p == 'none':
    for i,name in enumerate(node_name):
        if i == 0:
            print('input_name :',name)
        elif i == len(node_name)-1:
            print('output_name :',name)
        else:
            pass
else:
    for name in node_name:
        print(name)
    print('\n')
    for i,name in enumerate(node_name):
        if i == 0:
            print('input_name :',name)
        elif i == len(node_name)-1:
            print('output_name :',name)
        else:
            pass