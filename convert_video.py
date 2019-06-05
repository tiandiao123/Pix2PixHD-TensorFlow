import os
import argparse

# ff_to_vid(pred_folder, '%04d.png', ind_folder+'/y_pred.mp4', resolution=args.resize_h)



parser = argparse.ArgumentParser()
parser.add_argument('--pred_folder', type=str, default='reflect', help='the input image folder')
parser.add_argument('--sample_st', type=int, default=1, help='the start id of image folder')
parser.add_argument('--sample_ed', type=int, default=1, help='the end id of image folder')
parser.add_argument('--resolution', type=int, default=256, help='size of image')

args = parser.parse_args()



def ff_to_vid(in_folder, in_format, out_file, resolution=256):
    if resolution==256: 
        b_rate = 2000000 
    elif resolution==512:
        b_rate = 4000000
    else:
        raise NotImplementError('Resolution %d is not supported now.'%(resolution))


    print("hello")

    cmd = 'ffmpeg -i {}/{} -pix_fmt yuv420p -b:v {} {} -y'.format(in_folder, in_format, b_rate, out_file)
    os.system(cmd)


in_format = '%04d.png'

for folder_id in range(args.sample_st, args.sample_ed+1):
	folder_path = os.path.join(args.pred_folder, str(folder_id))
	in_folder = os.path.join(folder_path, "y_pred")

	ff_to_vid(in_folder, in_format, folder_path +"/y_pred.mp4", args.resolution)





