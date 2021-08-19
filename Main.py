import time
import ntpath
from PIL import Image
from Config import *
from torchvision import transforms
from Networks import EnlightenGAN_Network
from utils.utils import save_image
from utils import data_loader


def train():
    DataLoader = data_loader.DataLoader()
    dataset = DataLoader.load_data()
    GAN_Network = EnlightenGAN_Network.Network()
    # dataset_size = len(DataLoader)
    total_steps = 0

    for epoch in range(1, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        for i, data in enumerate(dataset):
            # iter_start_time = time.time()
            total_steps += opt.batchSize
            # epoch_iter = total_steps - dataset_size * (epoch - 1)
            GAN_Network.set_input(data)
            GAN_Network.optimize_parameters(epoch)

            # todo : Push the results to the Website
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
            GAN_Network.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

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


def predict(image_list: list, user_ip: str, isWeb=False):
    imgs = []
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    for image_path in image_list:
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
            save_image(image_numpy, image_dir, name, label)


if __name__ == '__main__':
    if opt.train:
        # todo: post them to the website
        train()
    elif opt.predict:
        # todo: post them to the website
        # if opt.use_models:
        #     image_dir = os.path.join(os.path.join(ROOT_PATH, 'Data'), 'Results')
        #     for iteration in range(5, 1000 + 5, 5):
        #         opt.which_epoch = str(iteration)
        #         if not os.path.exists(os.path.join(image_dir, '%s' % iteration)):
        #             os.mkdir(os.path.join(image_dir, '%s' % iteration))
        #         save_image_path = os.path.join(image_dir, '%s' % iteration)
        #         file_name = str(iteration) + '.txt'
        #         predict(image_dir=save_image_path, file_name=file_name)
        #     predict(image_dir=os.path.join(image_dir, 'Latest'))
        # else:
        #     predict()
        pass
