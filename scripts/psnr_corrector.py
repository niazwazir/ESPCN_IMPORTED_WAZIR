f = open("PSNR_value_list.log",'r')
for line in f:
    preprocess = line.split()
    preprocess[5] = str(float(preprocess[5]) * 200 / 30)
    