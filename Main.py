import time
import ntpath
from Config import *
from Networks import EnlightenGAN_Network
from utils.utils import save_image
from utils import data_loader


DataLoader = data_loader.DataLoader()
dataset = DataLoader.load_data()
GAN_Network = EnlightenGAN_Network.Network()


def train():
    dataset_size = len(DataLoader)
    total_steps = 0

    for epoch in range(1, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter = total_steps - dataset_size * (epoch - 1)
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


def predict(image_dir=os.path.join(ROOT_PATH, 'Data/Results/')):
    for i, data in enumerate(dataset):
        GAN_Network.set_input(data)
        visuals = GAN_Network.predict()
        img_path = GAN_Network.get_image_paths()
        print('process image... %s' % img_path)
        print('=' * 255)
        ims = []
        txts = []
        links = []
        short_path = ntpath.basename(img_path[0])
        name = os.path.splitext(short_path)[0]
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)
        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        # visualizer.save_images(webpage, visuals, img_path)


if __name__ == '__main__':
    if opt.train:
        # todo: post them to the website
        train()
    elif opt.predict:
        # todo: post them to the website
        if opt.use_models:
            image_dir = os.path.join(ROOT_PATH, 'Data/Results/')
            for iteration in range(5, 1000 + 5, 5):
                save_image_path = image_dir + str(iteration) + '/'
                predict(save_image_path)
            predict(image_dir+'Latest/')
        else:
            predict()
