import sys
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import cv2
random.seed(100)
class Gen_data_Meta(object):
    def __init__(self,data_folder,mode,max_num_pattern=5,num_in_and_out=50,num_in=10):
        self.data_folder = data_folder
        self.num_in_and_out=num_in_and_out
        self.max_num_pattern=max_num_pattern
        self.num_different_pattern=10
        self.mode=mode
        self.num_in = num_in
        self.dsize = (28,28)

        self.character_folders = [os.path.join(self.data_folder, family, character) \
                                  for family in os.listdir(self.data_folder) \
                                  if os.path.isdir(os.path.join(self.data_folder, family)) \
                                  for character in os.listdir(os.path.join(self.data_folder, family))]
        self.character_folders_train = random.sample(population=self.character_folders, k=1000)
        self.character_folders_test = [x for x in self.character_folders if x not in self.character_folders_train]

    def imgs_labels_train(self,paths, labels, num_sample):
        images_source1 = [(i, os.path.join(path, image)) for i, path in zip(labels, paths) for image in self.sampler(os.listdir(path),num_sample)]
        label_source1, img_source1_file = zip(*images_source1)
        img_source1 = np.array([cv2.resize(plt.imread(filename),dsize=self.dsize) for filename in img_source1_file])

        return img_source1, label_source1

    def imgs_labels(self,paths, labels, num_sample):
        images_source1 = [(i, os.path.join(path, image)) for i, path in zip(labels, paths) for image in
                          self.sampler(os.listdir(path), num_sample[0][i])]
        random.shuffle(images_source1)
        label_source1, img_source1_file = zip(*images_source1)

        img_source1 = np.array([cv2.resize(plt.imread(filename),dsize=self.dsize) for filename in img_source1_file])

        return img_source1, label_source1


    def sampler(self,x,num_in):
        return list(np.random.choice(x, num_in))

    def construction_unknown_single_train(self):
        if self.mode == 'train':
            folder_sel = (random.sample(self.character_folders_train, 1 + self.num_different_pattern))
        if self.mode == 'test':
            folder_sel = (random.sample(self.character_folders_test, 1 + self.num_different_pattern))
        folder_sel_in = folder_sel[:1]
        folder_sel_out = folder_sel[1:]

        num_in_and_out_half = np.int(self.num_in_and_out / 2.0)
        num_sample_sel_out = np.random.multinomial(self.num_in_and_out - num_in_and_out_half - self.num_different_pattern,
                                                        [float(1 / (self.num_different_pattern))] * (
                                                            self.num_different_pattern),
                                                        size=1) + 1
        in_labels = range(1)
        out_labels = range(self.num_different_pattern)

        img_in1, _ = self.imgs_labels_train(folder_sel_in, in_labels, self.num_in)
        img_in2, _ = self.imgs_labels_train(folder_sel_in, out_labels, num_in_and_out_half)

        img_out, _ = self.imgs_labels(folder_sel_out, out_labels,num_sample_sel_out)
        img_in_and_out = np.concatenate((img_in2, img_out), axis=0)
        lab_in_and_out = np.array([0.0] * num_in_and_out_half + [1.0] * (self.num_in_and_out - num_in_and_out_half))

        return img_in1, img_in_and_out, lab_in_and_out

    def construction_unknown_single_test(self, num_pattern_sel):
        if self.mode == 'train':
            folder_sel = (random.sample(self.character_folders_train, num_pattern_sel + self.num_different_pattern))
        if self.mode == 'test':
            folder_sel = (random.sample(self.character_folders_test, num_pattern_sel + self.num_different_pattern))

        folder_sel_known = folder_sel[:num_pattern_sel]
        folder_sel_unknown = folder_sel[num_pattern_sel:]

        num_in_and_out_half = np.int(np.around(self.num_in_and_out / 2.0))

        num_sample_sel_out = np.random.multinomial(self.num_in_and_out - num_in_and_out_half - self.num_different_pattern,
                                                        [float(1 / (self.num_different_pattern))] * (
                                                            self.num_different_pattern),
                                                        size=1) + 1
        in_labels = range(num_pattern_sel)
        out_labels = range(self.num_different_pattern)

        img_in1, label_in1 = self.imgs_labels_train(folder_sel_known, in_labels, self.num_in)
        img_in2, label_in2 = self.imgs_labels_train(folder_sel_known, in_labels, self.num_in)

        img_out, _ = self.imgs_labels(folder_sel_unknown, out_labels,
                                              num_sample_sel_out)

        img_in_and_out = np.concatenate((img_in2, img_out), axis=0)
        lab_in_and_out = np.array([0.0] * num_in_and_out_half + [1.0] * (self.num_in_and_out - num_in_and_out_half))

        return img_in1, np.array(label_in1), img_in_and_out, lab_in_and_out, np.array(label_in2)


