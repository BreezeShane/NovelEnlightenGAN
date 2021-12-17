import time
import ntpath
from Config import *
from PIL import Image
from torchvision import transforms
from utils import data_loader, utils, preprocessor
from Networks import EnlightenGAN_Network
from torch.utils.tensorboard import SummaryWriter


def train(mode: int):
    DataLoader = data_loader.DataLoader()
    dataset = DataLoader.load_data()
    GAN_Network = EnlightenGAN_Network.Network()
    # dataset_size = len(DataLoader)
    total_steps = 0

    scratch_line = None
    if mode == 1:
        scratch_line = int(opt.which_epoch)
    elif mode == 0:
        scratch_line = 0

    for epoch in range(scratch_line + 1, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        for i, data in enumerate(dataset):
            # iter_start_time = time.time()
            total_steps += opt.batchSize
            # epoch_iter = total_steps - dataset_size * (epoch - 1)
            GAN_Network.set_input(data)
            GAN_Network.optimize_parameters(epoch)

            # if total_steps % opt.display_freq == 0:
            #     visualizer.display_current_results(GAN_Network.get_current_visuals(), epoch)

            # if total_steps % opt.print_freq == 0:
            #     errors = GAN_Network.get_current_errors(epoch)
            #     t = (time.time() - iter_start_time) / opt.batchSize
            #     # visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            #     # if opt.display_id > 0:
            #     #     visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest GAN_Network (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                GAN_Network.save('latest')

        if epoch % opt.save_epoch_freq == 0:
            print('saving the GAN_Network at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            GAN_Network.save('latest')
            if epoch >= 100:
                GAN_Network.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        # ~~~ The next part is added by hand. In order to record the whole process. ~~~
        Writer = SummaryWriter(log_dir=os.path.join(SAVE_ROOT_PATH, 'log'), comment='Loss Group')
        # print('The loss values are: ')
        network_errors = GAN_Network.get_current_errors(epoch)
        with Writer as wrt:
            wrt.add_scalars('Loss Group', dict(network_errors), epoch)
        if epoch >= 100:
            current_images = GAN_Network.get_current_visuals()
            save_images_path = os.path.join(SAVE_ROOT_PATH, 'Processing', str(epoch))
            if not os.path.exists(save_images_path):
                os.mkdir(save_images_path)
            for image_key in current_images.keys():
                utils.save_image(current_images[image_key], save_images_path, epoch, image_key, isWeb=False)
        # ~~~ The part I add is ended. ~~~

        if opt.new_lr:
            if epoch == opt.niter:
                GAN_Network.update_learning_rate()
            elif epoch == (opt.niter + 20):
                GAN_Network.update_learning_rate()
            elif epoch == (opt.niter + 70):
                GAN_Network.update_learning_rate()
            elif epoch == (opt.niter + 90):
                GAN_Network.update_learning_rate()
                GAN_Network.update_learning_rate()
                GAN_Network.update_learning_rate()
                GAN_Network.update_learning_rate()
        else:
            if epoch > opt.niter:
                GAN_Network.update_learning_rate()


def predict(image_path_list: list, user_ip: str, isWeb=False, save_dir=''):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img_processor = preprocessor.PreProcessor(channels=3)
    GAN_Network = EnlightenGAN_Network.Network()
    for image_path in image_path_list:
        img_patch_list = []
        img_processor.clear_container()
        img = Image.open(image_path).convert('RGB')
        img_patch_list = img_processor(img)
        img_patches_with_gray_and_path = []
        for patch_img in img_patch_list:
            patch_img = transform(patch_img)
            img_input = torch.unsqueeze(patch_img, 0)
            r, g, b = patch_img[0] + 1, patch_img[1] + 1, patch_img[2] + 1
            gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.
            gray = torch.unsqueeze(gray, 0).unsqueeze(0)
            patch_img_with_gray_and_path = [img_input, gray, image_path]
            img_patches_with_gray_and_path.append(patch_img_with_gray_and_path)
        restore_imgs_list = []
        label = name = None
        output_info = True
        for data in img_patches_with_gray_and_path:
            GAN_Network.set_input_single(data)
            visuals = GAN_Network.predict()
            if output_info:
                print('process image %s' % data[2])
                short_path = ntpath.basename(data[2])
                name = os.path.splitext(short_path)[0]
                output_info = False
            for label, image_numpy in visuals.items():
                restore_imgs_list.append(image_numpy)
        img_processor.clear_container()
        for item in restore_imgs_list:
            img_processor.add_to_container(item)
        restored_img = img_processor.restore_picture(output_type='rgb')
        if not opt.use_models:
            if isWeb:
                image_root_path = os.path.join(ROOT_PATH, 'front-end', 'Static_Files', 'downloads', user_ip)
            else:
                image_root_path = os.path.join(ROOT_PATH, 'Data', 'Result')
        else:
            image_root_path = os.path.join(save_dir)
        utils.save_image(restored_img, image_root_path, name, label, isWeb)


if __name__ == '__main__':
    if opt.train:
        train(mode=0)
    elif opt.predict:
        image_dir = os.path.join(ROOT_PATH, 'Data', 'Test_data')
        image_result_dir = os.path.join(ROOT_PATH, 'Data', 'Result')
        image_path_list = []
        for file_name in os.listdir(image_dir):
            file_path = os.path.join(image_dir, file_name)
            image_path_list.append(file_path)
        if opt.use_models:
            for iteration in range(100, 500 + 5, 5):
                opt.which_epoch = str(iteration)
                save_image_path = os.path.join(image_result_dir, '%s' % iteration)
                if not os.path.exists(save_image_path):
                    os.mkdir(save_image_path)
                predict(image_path_list=image_path_list, user_ip="", isWeb=False, save_dir=save_image_path)
        else:
            predict(image_path_list=image_path_list, user_ip="", isWeb=False)
    elif opt.continue_train:
        print('Where do you want to start ?')
        print('A. From epoch one.')
        print('B. From last epoch.')

        which_mode = input()
        if which_mode.upper() == 'A':
            train(mode=0)
        elif which_mode.upper() == 'B':
            train(mode=1)
