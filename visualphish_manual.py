import visualphish_main
import visualphish_manual
import os
import argparse
import sys


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-f', "--folder", help='Input the folder of files to parse', required=True)
    # parser.add_argument('-r', "--results", help='Input the folder of files to parse', required=True)
    # args = parser.parse_args()
    # path = args.folder
    # if not os.path.exists(args.results):
    #     os.makedirs(args.results)
    # results = os.path.join(args.results, 'results.txt')

    visualphish_main.main(r'C:\Users\вадим\VScode\VisualPhishNet\test', r'C:\Users\вадим\VScode\VisualPhishNet\results313.txt')

    
    def image_bytearray(path):
        with open(path, "rb") as image:
            f = image.read()
            b = bytearray(f)
        return b
    
    image = image_bytearray('test/shot.png')
    visualphish_main.result_without_dir(image)
    ## Example: python visualphish_manual.py --folder ..\\backup\\DATABASE\\260720 --results .



