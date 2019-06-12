import os
import numpy as np 
import tensorflow as tf
import argparse
import pathlib
import time 
import glob
from data_generator import load_images_paired2
import scipy
import warnings
warnings.filterwarnings('ignore')

tf.reset_default_graph()

curr_path = os.getcwd()

parser = argparse.ArgumentParser()

# params
parser.add_argument('--resize_w', type=int, default=256)
parser.add_argument('--resize_h', type=int, default=256)
parser.add_argument('--frame_count', type=int, default=11)
parser.add_argument('--batch_size', type=int, default=2, help='gen and disc batch size')
parser.add_argument('--crop_h_flag', type=int, default=0, help='0 : pos(420, 0) scale(720, 720), 10: args.crop_pos args.crop_scale, 20: json auto')
parser.add_argument('--no_crop', action='store_true', help='no crop')
parser.add_argument('--img_width', type=int, default=720)
parser.add_argument('--img_height', type=int, default=1280)
parser.add_argument('--need_resize', type=bool, default=False)
parser.add_argument("--mode", dest='mode', type=str, help="specify the checkpoint")


# model 
parser.add_argument("--checkpoint_name", dest='checkpoint_name', type=str, help="specify the checkpoint")
parser.add_argument("--model", dest='model', type=str, help="specify the model: non_quant, quant")
parser.add_argument('--model_name', type=str, default='auto', help='folder name to save weights')
parser.add_argument('--epoch_id_inference', type=int, default=9)

# dataset 
parser.add_argument('--dataroot', type=str, default='./datasets/', help='path to training data')
parser.add_argument('--infer_out_root', type=str, default='./infer_res', help='results root folder of inference')
parser.add_argument('--sample_st', type=int, default=-1)
parser.add_argument('--sample_ed', type=int, default=-1)


args = parser.parse_args()


def findAllTuple(args):
    images_count = {}
    for i in range(args.sample_st, args.sample_ed+1):
        folder_id = i
        folder_path = os.path.join(args.dataroot, str(folder_id))
        sub_folder_path = os.path.join(folder_path, 'PNCC')
        images = glob.glob(sub_folder_path+"/*.png")
        count_len = len(images)
        images_count[i] = count_len

    train_list = []
    for folder_id in range(args.sample_st, args.sample_ed+1):
    	for image_index in range((images_count[folder_id]-args.frame_count)):
            image_id = image_index + 1
            train_list.append([folder_id, image_id, image_id+args.frame_count-1])
    			
    return train_list


def get_inference_tuples(folder_id):
    folder_path = os.path.join(args.dataroot, str(folder_id))
    sub_folder_path = os.path.join(folder_path, 'PNCC')
    images = glob.glob(sub_folder_path+"/*.png")

    count_len = len(images)
    test_list = []
    for image_index in range(count_len-args.frame_count):
        image_id = image_index+1
        test_list.append((folder_id, image_id, image_id+args.frame_count-1))
    

    return test_list


def ckpt_to_pb(args):
    args.epoch_id_inference = 9
    args.model_name = 'model256_denseposeV4_mse_0610'
    args.checkpoint_name = './checkpoints/model256_denseposeV4_mse_0610/'

    # restore from meta graph (no need to re-build model)
    print ("Restore model_epoch{}.ckpt.meta from {}".format(str(args.epoch_id_inference), args.checkpoint_name))
    graph_file_path = os.path.join(args.checkpoint_name, "model_epoch{}.ckpt.meta".format(str(args.epoch_id_inference)))
    saver = tf.train.import_meta_graph(graph_file_path)

    # Config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    # show number of paramaters 
    total_param= np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])/1e6
    print ('Total params: %.1fM'%(total_param))

    infer_out_root = os.path.join(args.infer_out_root, args.model_name)
    if not os.path.isdir(infer_out_root):
        os.mkdir(infer_out_root)

    print("Converting model from *.ckpt to *.pb...")

    with tf.Session(config=config) as sess: 
        with tf.device('/gpu:0'):
            # Initialize all variables and start queue runners  
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            
            # To continue training from one of the checkpoints
            if not args.checkpoint_name:
                raise IOError('In test mode, a checkpoint is expected.')
                
            saver.restore(sess, args.checkpoint_name + 'model_epoch%d.ckpt'%args.epoch_id_inference)
            
            #gets a reference to the list containing the trainable variables
            trainable_collection = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
            variables_to_remove = [v for v in tf.trainable_variables() if v.name.startswith("dis_")]
            for rem in variables_to_remove:
                trainable_collection.remove(rem)
            
            generator_image =  tf.get_default_graph().get_tensor_by_name('Tanh:0')
            image_A = tf.get_default_graph().get_tensor_by_name('Placeholder:0')
            #image_B = tf.get_default_graph().get_tensor_by_name('Placeholder_1:0')

            save_pb_model_dir = os.path.join(args.checkpoint_name, 'save_model_pb_epoch%d'%args.epoch_id_inference)
            if not os.path.isdir(save_pb_model_dir):
                os.mkdir(save_pb_model_dir)
        
            # check if file exist 
            if not os.path.isfile(save_pb_model_dir + '/saved_model.pb'):
                print ('saving new model in .pb ......')
                tf.saved_model.simple_save(sess, save_pb_model_dir, inputs={"myInput1": image_A}, outputs={"myOutput": generator_image})
            else:
                print ('.pb model found!!!')

def pb_to_tflite(args):
    
    print ('Converting *.pd to *.tflite...')
    save_pb_model_dir = os.path.join(args.checkpoint_name, 'save_model_pb_epoch%d'%args.epoch_id_inference)
    
    if not os.path.isfile(save_pb_model_dir + "/tflite_models/model_original_epoch%d.tflite"%args.epoch_id_inference):
        
        # find the model 
        saved_model_dir = str(sorted(pathlib.Path(args.checkpoint_name).glob("*"))[-1])
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        tflite_model = converter.convert()
        
         # save orginal model 
        tflite_models_dir = pathlib.Path(saved_model_dir + "/tflite_models/")
        tflite_models_dir.mkdir(exist_ok=True, parents=True)
        tflite_model_file = tflite_models_dir/("model_original_epoch%d.tflite"%args.epoch_id_inference)
        tflite_model_file.write_bytes(tflite_model)
        
        # Note: If you don't have a recent tf-nightly installed, the
        # "optimizations" line will have no effect.
        tf.logging.set_verbosity(tf.logging.INFO)
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        tflite_quant_model = converter.convert()
        tflite_model_quant_file = tflite_models_dir/("model_quantized_8bit_epoch%d.tflite"%args.epoch_id_inference)
        tflite_model_quant_file.write_bytes(tflite_quant_model)

    else:
        print ('.tflite model found!!!')


def inference_lite(args):
    image_size = args.resize_w
    frame_count = args.frame_count
    tflite_models_dir = pathlib.Path(args.checkpoint_name + "/save_model_pb_epoch%d/tflite_models/"%args.epoch_id_inference)
    tflite_model_file = tflite_models_dir/("model_original_epoch%d.tflite"%args.epoch_id_inference) if args.model=='non_quant' else tflite_models_dir/("model_quantized_8bit_epoch%d.tflite"%args.epoch_id_inference)

    #print ('load model from', tflite_model_file)
    
    # load model 
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
    interpreter.allocate_tensors()
    
    input_index1 = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    
    infer_out_root = os.path.join(args.infer_out_root, args.model_name)
    if not os.path.isdir(infer_out_root):
        os.mkdir(infer_out_root)
    
    # do inference
    start_time = time.time()
    count = 0     
    for folder_id in range(args.sample_st, args.sample_ed+1):
        test_list = get_inference_tuples(folder_id)
        ind_folder = os.path.join(infer_out_root, str(folder_id))

        # mkdir 
        if not os.path.isdir(ind_folder):
            os.mkdir(ind_folder)
            
        pred_folder = os.path.join(ind_folder, 'y_pred_%s'%(args.model))
        if not os.path.isdir(pred_folder):
            os.mkdir(pred_folder)
            
        for iter_id in np.arange(0, len(test_list), args.batch_size):
            if iter_id+args.batch_size-1>=len(test_list):
                break
            batch_A, batch_B = load_images_paired2(args, iter_id, args.batch_size, image_size, frame_count, test_list)
            
            # inference 
            interpreter.set_tensor(input_index1, batch_A.astype(np.float32))

            fake_B = interpreter.get_tensor(output_index)
            count+=2
            
#            for inner_id in range(args.batch_size):
#                count+=1
#                fake_B = np.array(fake_B)
#                              
#                unstack_fake_B = fake_B[inner_id]
#                unstack_fake_B = unstack_fake_B[:,:,(2,1,0)]
#        
#                img1 = scipy.misc.toimage((unstack_fake_B+1)*127.5)
#                img1.save(pred_folder+'/%04d.png'%(count))
        
        
        time_cost = time.time() - start_time
        print ('[%s] Processed %d images in %.3fs . FPS=%.2f'%(args.model, count, time_cost, count/time_cost))
        interpreter.invoke()
     
        
def inference(args):
    tf.reset_default_graph()
    image_size = args.resize_w
    frame_count = args.frame_count
    #print ("model_epoch{}.ckpt.meta".format(str(args.epoch_id_inference)))
    graph_file_path = os.path.join(args.checkpoint_name, "model_epoch{}.ckpt.meta".format(str(args.epoch_id_inference)))
    saver = tf.train.import_meta_graph(graph_file_path)
    
    # Config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    infer_out_root = os.path.join(args.infer_out_root, args.model_name)
    if not os.path.isdir(infer_out_root):
        os.mkdir(infer_out_root)

    with tf.Session(config=config) as sess: 
        with tf.device('/gpu:0'):
            # Initialize all variables and start queue runners  
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            
            # To continue training from one of the checkpoints
            if not args.checkpoint_name:
                raise IOError('In test mode, a checkpoint is expected.')
            module_file = tf.train.latest_checkpoint(args.checkpoint_name)
#            print (module_file)
#            module_file = './checkpoints/model256_denseposeV4_mse/model_epoch9.ckpt'
            saver.restore(sess, module_file)

            generator_image =  tf.get_default_graph().get_tensor_by_name('Tanh:0')
            image_A = tf.get_default_graph().get_tensor_by_name('Placeholder:0')
            image_B = tf.get_default_graph().get_tensor_by_name('Placeholder_1:0')
            G_loss = tf.get_default_graph().get_tensor_by_name('G_loss_GAN:0')


            # Test network
            start_time = time.time()
            count = 0
            for folder_id in range(args.sample_st, args.sample_ed+1):

                test_list = get_inference_tuples(folder_id)
                
                # mkdir 
                ind_folder = os.path.join(infer_out_root, str(folder_id))
                if not os.path.isdir(ind_folder):
                    os.mkdir(ind_folder)
                pred_folder = os.path.join(ind_folder, 'y_pred_sess')
                if not os.path.isdir(pred_folder):
                    os.mkdir(pred_folder)

                for iter_id in np.arange(0, len(test_list), args.batch_size):
                    if iter_id+args.batch_size-1>=len(test_list):
                        break
                    batch_A, batch_B = load_images_paired2(args, iter_id, args.batch_size, image_size, frame_count, test_list)
                        
                    fake_B, cur_g_loss = sess.run([generator_image, G_loss], \
                        feed_dict={image_A: batch_A.astype('float32'), image_B: batch_B.astype('float32')})
                    count+=2
                    
#                    for inner_id in range(args.batch_size):
#                        count+=1
#                        fake_B = np.array(fake_B)
#
#                        unstack_fake_B = fake_B[inner_id]
#                        unstack_fake_B = unstack_fake_B[:,:,(2,1,0)]
#
#                        img1 = scipy.misc.toimage((unstack_fake_B+1)*127.5)
#                        img1.save(pred_folder+'/%04d.png'%(count))

            time_cost = time.time() - start_time
            print ('[%s] Processed %d images in %.3fs . FPS=%.2f'%(args.model, count, time_cost, count/time_cost))
        

# ckpt to pb
args.epoch_id_inference = 9
args.model_name = 'model256_denseposeV4_mse_0610'
args.checkpoint_name = './checkpoints/model256_denseposeV4_mse_0610/'
ckpt_to_pb(args)


# pb to tflite
args.epoch_id_inference = 9
args.model_name = 'model256_denseposeV4_mse_0610'
args.checkpoint_name = './checkpoints/model256_denseposeV4_mse_0610/'
pb_to_tflite(args)


# inference test
args.mode = 'test'
args.dataroot = './datasets/'
args.batch_size = 2
args.frame_count = 11
args.sample_st = 80
args.sample_ed = 80

args.model = 'non_quant'
inference_lite(args)

args.model = 'quant'
inference_lite(args)
    
args.model = 'sess'
inference(args)
