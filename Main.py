import time
import ntpath
from Config import *
from PIL import Image
from torchvision import transforms
from utils import data_loader, utils
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
        scratch_line = opt.which_epoch
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

        if epoch % opt.save_epoch_freq == 0 and epoch >= 100:
            print('saving the GAN_Network at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            GAN_Network.save('latest')
            GAN_Network.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        # ~~~ The next part is added by hand. In order to record the whole process. ~~~
        # writers = [
        #     SummaryWriter(log_dir=os.path.join(save_root_path, 'log', 'D_A'), comment='Global Discriminator'),
        #     SummaryWriter(log_dir=os.path.join(save_root_path, 'log', 'G_A'), comment='Global Generator'),
        #     SummaryWriter(log_dir=os.path.join(save_root_path, 'log', 'VGG'), comment='Perceptual Extractor'),
        #     SummaryWriter(log_dir=os.path.join(save_root_path, 'log', 'D_P'), comment='Local Discriminator')
        # ]
        Writer = SummaryWriter(log_dir=os.path.join(save_root_path, 'log'), comment='Loss Group')
        # print('The loss values are: ')
        network_errors = GAN_Network.get_current_errors(epoch)
        for scalar in network_errors.items():
            with Writer as wrt:
                wrt.add_scalars(scalar[0], dict(scalar), epoch)
            # print(loss, ' Loss at epoch ', epoch, ' is ', network_errors[loss])
        if epoch >= 100:
            current_images = GAN_Network.get_current_visuals()
            save_images_path = os.path.join(save_root_path, 'Processing', str(epoch))
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


def predict(image_path_list: list, user_ip: str, isWeb=False):
    imgs = []
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    for image_path in image_path_list:
        img = Image.open(image_path).convert('RGB')
        img = transform(img)
        img_input = torch.unsqueeze(img, 0)
        r, g, b = img[0] + 1, img[1] + 1, img[2] + 1
        gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.
        gray = torch.unsqueeze(gray, 0).unsqueeze(0)
        img_with_gray_and_path = [img_input, gray, image_path]
        imgs.append(img_with_gray_and_path)
    GAN_Network = EnlightenGAN_Network.Network()
    for data in imgs:
        GAN_Network.set_input_single(data)
        visuals = GAN_Network.predict()
        print('process image %s' % data[2])
        short_path = ntpath.basename(data[2])
        name = os.path.splitext(short_path)[0]
        if isWeb:
            image_dir = os.path.join(ROOT_PATH, 'front-end', 'Static_Files', 'downloads', user_ip)
        else:
            image_dir = os.path.join(ROOT_PATH, 'Data', 'Result')
        for label, image_numpy in visuals.items():
            utils.save_image(image_numpy, image_dir, name, label)


if __name__ == '__main__':
    if opt.train:
        train(mode=0)
    elif opt.predict:
        image_dir = os.path.join(os.path.join(ROOT_PATH, 'Data'), 'Test_data')
        image_path_list = []
        for file_name in os.listdir(image_dir):
            file_path = os.path.join(image_dir, file_name)
            image_path_list.append(file_path)
        if opt.use_models:
            for iteration in range(100, 500 + 5, 5):
                opt.which_epoch = str(iteration)
                save_image_path = os.path.join(image_dir, '%s' % iteration)
                if not os.path.exists(save_image_path):
                    os.mkdir(save_image_path)
                predict(image_path_list=image_path_list, user_ip="", isWeb=False)
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
