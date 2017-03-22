#!/bin/bash

#python3 project3.py -i 20170316 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y5 data/y6 data/y7 data/y8 data/y9 -m resnet_ish -b 32 -e 20 -d 0.5 0.2 0.1
#python3 project3.py -i 20170316 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y5 data/y6 data/y7 data/y8 data/y9 -m end_to_end_nvidia -b 32 -e 20 -d 0.5 0.2 0.1
#python3 project3.py -i 20170316 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y5 data/y6 data/y7 data/y8 data/y9 -m conv_3_fc_3_more_filters -b 32 -e 20 -d 0.5 0.2
#python3 project3.py -i 20170316 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y5 data/y6 data/y7 data/y8 data/y9 -m conv_3_fc_3 -b 32 -e 20 -d 0.5 0.2

#python3 project3.py -i 20170317.03 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m vgg16_ish -b 32 -e 30 -d 0.5 0.2 0.1
#python3 project3.py -i 20170317.03 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m resnet_ish -b 32 -e 30 -d 0.5 0.2 0.1
#python3 project3.py -i 20170317.03 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m end_to_end_nvidia -b 32 -e 30 -d 0.5 0.2 0.1
#python3 project3.py -i 20170317.03 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m conv_3_fc_3_more_filters -b 32 -e 30 -d 0.5 0.2
#python3 project3.py -i 20170317.03 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m conv_3_fc_3 -b 32 -e 30 -d 0.5 0.2

### Try changing the steering correction to 0.2
#python3 project3.py -i 20170319.02 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m end_to_end_nvidia -b 32 -x 0.2 -e 30 -d 0.5 0.5 0.5
#python3 project3.py -i 20170319.02 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m conv_4_fc_3_more_filters -b 32 -x 0.2 -e 30 -d 0.5 0.5
#python3 project3.py -i 20170319.02 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m conv_4_fc_3 -b 32 -x 0.2 -e 30 -d 0.5 0.5
#
### Ignore 80% of all samples where the steering angle is 0.0
#python3 project3.py -i 20170319.02 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m end_to_end_nvidia -b 32 -z 0.8 -e 30 -d 0.5 0.5 0.5
#python3 project3.py -i 20170319.02 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m conv_4_fc_3_more_filters -b 32 -z 0.8 -e 30 -d 0.5 0.5
#python3 project3.py -i 20170319.02 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m conv_4_fc_3 -b 32 -z 0.8 -e 30 -d 0.5 0.5
#
### Increase dropout
#python3 project3.py -i 20170319.02 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m end_to_end_nvidia -b 32 -e 30 -d 0.75 0.63 0.5
#python3 project3.py -i 20170319.02 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m conv_4_fc_3_more_filters -b 32 -e 30 -d 0.75 0.63
#python3 project3.py -i 20170319.02 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m conv_4_fc_3 -b 32 -e 30 -d 0.75 0.63

### Try changing the steering correction to 0.15 and ignoring 90% of zero angle samples
#python3 project3.py -i 20170320.01 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m end_to_end_nvidia -b 32 -z 0.9 -x 0.15 -e 30 -d 0.5 0.5 0.5
#python3 project3.py -i 20170320.01 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m conv_4_fc_3_more_filters -b 32 -z 0.9 -x 0.15 -e 30 -d 0.5 0.5
#python3 project3.py -i 20170320.01 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m conv_4_fc_3 -b 32 -z 0.9 -x 0.15 -e 30 -d 0.5 0.5
#
### Try changing the steering correction to 0.2 and ignoring 80% of zero angle samples
#python3 project3.py -i 20170320.01 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m end_to_end_nvidia -b 32 -z 0.8 -x 0.2 -e 30 -d 0.5 0.5 0.5
#python3 project3.py -i 20170320.01 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m conv_4_fc_3_more_filters -b 32 -z 0.8 -x 0.2 -e 30 -d 0.5 0.5
#python3 project3.py -i 20170320.01 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m conv_4_fc_3 -b 32 -z 0.8 -x 0.2 -e 30 -d 0.5 0.5
#
### Use only center images
#python3 project3.py -i 20170320.01 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m end_to_end_nvidia -b 32 -c -e 30 -d 0.5 0.5 0.5
#python3 project3.py -i 20170320.01 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m conv_4_fc_3_more_filters -b 32 -c -e 30 -d 0.5 0.5
#python3 project3.py -i 20170320.01 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m conv_4_fc_3 -b 32 -c -e 30 -d 0.5 0.5

### Try changing the steering correction to 0.1 and ignoring 90% of zero angle samples
#python3 project3.py -i 20170320.01 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m end_to_end_nvidia -b 32 -z 0.9 -e 30 -d 0.5 0.5 0.5
#python3 project3.py -i 20170320.01 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m conv_4_fc_3_more_filters -b 32 -z 0.9 -e 30 -d 0.5 0.5
#python3 project3.py -i 20170320.01 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m conv_4_fc_3 -b 32 -z 0.9 -e 30 -d 0.5 0.5
#
### Try changing the steering correction to 0.1 and ignoring 80% of zero angle samples and dropout 0.63 0.63 0.63
#python3 project3.py -i 20170320.01 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m end_to_end_nvidia -b 32 -z 0.8 -e 30 -d 0.63 0.63 0.63
#python3 project3.py -i 20170320.01 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m conv_4_fc_3_more_filters -b 32 -z 0.8 -e 30 -d 0.63 0.63
#python3 project3.py -i 20170320.01 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m conv_4_fc_3 -b 32 -z 0.8 -e 30 -d 0.63 0.63
#
### Try changing the steering correction to 0.1 and ignoring 80% of zero angle samples and dropout 0.37 0.37 0.37
#python3 project3.py -i 20170320.01 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m end_to_end_nvidia -b 32 -z 0.8 -e 30 -d 0.37 0.37 0.37
#python3 project3.py -i 20170320.01 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m conv_4_fc_3_more_filters -b 32 -z 0.8 -e 30 -d 0.37 0.37
#python3 project3.py -i 20170320.01 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m conv_4_fc_3 -b 32 -z 0.8 -e 30 -d 0.37 0.37

### Try changing the steering correction to 0.1 and ignoring 80% of zero angle samples and dropout 0.5 0.5 0.5
#python3 project3.py -i 20170321.01 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m end_to_end_nvidia -b 32 -z 0.8 -e 30 -d 0.5 0.5 0.5
#python3 project3.py -i 20170321.01 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m conv_4_fc_3_more_filters -b 32 -z 0.8 -e 30 -d 0.5 0.5
#python3 project3.py -i 20170321.01 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m conv_4_fc_3 -b 32 -z 0.8 -e 30 -d 0.5 0.5

## Try changing the steering correction to 0.1 and ignoring 80% of zero angle samples and dropout 0.63 0.63 0.63
python3 project3.py -i 20170321.02 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m end_to_end_nvidia -b 32 -z 0.8 -e 30 -d 0.63 0.63 0.63
python3 project3.py -i 20170321.02 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m conv_4_fc_3_more_filters -b 32 -z 0.8 -e 30 -d 0.63 0.63
python3 project3.py -i 20170321.02 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m conv_4_fc_3 -b 32 -z 0.8 -e 30 -d 0.63 0.63

## Try changing the steering correction to 0.1 and ignoring 80% of zero angle samples and dropout 0.75 0.75 0.75
python3 project3.py -i 20170321.02 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m end_to_end_nvidia -b 32 -z 0.8 -e 30 -d 0.75 0.75 0.75
python3 project3.py -i 20170321.02 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m conv_4_fc_3_more_filters -b 32 -z 0.8 -e 30 -d 0.75 0.75
python3 project3.py -i 20170321.02 -s data/y0 data/y1 data/y2 data/y3 data/y4 data/y6 data/y7 data/y8 data/y9 -m conv_4_fc_3 -b 32 -z 0.8 -e 30 -d 0.75 0.75

