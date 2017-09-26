import caffe

if __name__ == "__main__":
    root = '/home/hans/caffe/models/bvlc_alexnet/'
    #caffe.set_mode_cpu
    net = caffe.Net(root+'deploy.prototxt',\
    root+'bvlc_alexnet.caffemodel',caffe.TEST)    
    conv_w = []
    conv_b = []
    fc_w = []
    fc_b = []
    conv_max = []
    conv_min = []
    fc_max = []
    fc_min = []
    for i in range(1,6):
        conv_w.append(net.params['conv'+str(i)][0].data)
        conv_b.append(net.params['conv'+str(i)][1].data)
        conv_max.append(max(conv_w[i-1].max(),conv_b[i-1].max()))
        conv_min.append(min(conv_w[i-1].min(),conv_b[i-1].min()))
    for i in range(6,9):
        fc_w.append(net.params['fc'+str(i)][0].data)
        fc_b.append(net.params['fc'+str(i)][1].data)
        fc_max.append(max(fc_w[i-6].max(),fc_b[i-6].max()))
        fc_min.append(min(fc_w[i-6].min(),fc_b[i-6].min()))
    #for i in conv1_w.flat:
    #    conv1_w.flat = 0
    #net.save('/home/hans/caffe/models/bvlc_alexnet/new.caffemodel')
    #print net.params['conv1'][0].data[0,0,0,0]
    print conv_w[0]
    print conv_b[0]
    for i in range(5):
        print 'conv%d:' % (i+1)
        print '  ', conv_w[i].size, ' ', conv_b[i].size, ' ',\
        conv_max[i], ' ', conv_min[i]
    for i in range(3):
        print 'fc%d:'%(i+6)
        print '  ', fc_w[i].size, ' ', fc_b[i].size, ' ', fc_max[i],\
        ' ', fc_min[i]
    print 'total:'
    print '  ',sum([conv_w[0].size,conv_w[1].size,conv_w[2].size,conv_w[3].size,conv_w[4].size])\
    +sum([fc_w[0].size, fc_w[1].size, fc_w[2].size]),\
    sum([conv_b[0].size,conv_b[1].size,conv_b[2].size,conv_b[3].size,conv_b[4].size])\
    +sum([fc_b[0].size, fc_b[1].size, fc_b[2].size])

"""
import _init_paths
import caffe.proto.caffe_pb2 as caffe_pb2

caffemodel_filename = '/home/hans/caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel'

caffe_model = caffe_pb2.NetParameter()
try:
    f = open(caffemodel_filename, "rb")
    caffe_model.ParseFromString(f.read())
    f.close()
except IOError, e:
    print(caffemodel_filename+": file not opened. Create a new file.")
    pass
input("read done.")
"""
