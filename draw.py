﻿from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from google.protobuf import text_format

import caffe
import caffe.draw
from caffe.proto import caffe_pb2


def parse_args():
    """Parse input arguments
    """

    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('input_net_proto_file',
                        help='Input network prototxt file',
                        default='teraOGAN.prototxt')
    parser.add_argument('output_image_file',
                        help='Output image file',
                        default='teraGAN.png')
    parser.add_argument('--rankdir',
                        help=('One of TB (top-bottom, i.e., vertical), '
                              'RL (right-left, i.e., horizontal), or another '
                              'valid dot option; see '
                              'http://www.graphviz.org/doc/info/'
                              'attrs.html#k:rankdir'),
                        default='LR')
    parser.add_argument('--phase',
                        help=('Which network phase to draw: can be TRAIN, '
                              'TEST, or ALL.  If ALL, then all layers are drawn '
                              'regardless of phase.'),
                        default="ALL")

    args = parser.parse_args()
    return args


def main():
    #args = parse_args()
    net = caffe_pb2.NetParameter()
    text_format.Merge(open('teraGENGAN.prototxt').read(), net)
    print('Drawing net to %s' % 'teraGENGAN.png')
    phase=None;
    if False:
        if args.phase == "TRAIN":
            phase = caffe.TRAIN
        elif args.phase == "TEST":
            phase = caffe.TEST
        elif args.phase != "ALL":
            raise ValueError("Unknown phase: " + args.phase)
    caffe.draw.draw_net_to_file(net, 'teraGENGAN.png', 'LR',
                                phase)


if __name__ == '__main__':
    main()