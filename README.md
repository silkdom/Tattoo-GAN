# Tattoo-GAN
## Overview

Using GANs to generate tattoo design ideas. This project uses custom built notebooks to compile a imageset of tattoo flash illustration done by probable tattoo artists. From this imageset, generative adversarial network models (DCGAN & StyleGAN2) are trained and then tested to generate unique tattoo designs. Finally, a Colab notebook is built for simple and quick generation of tattoo design ideas!

<p align="center">
  <img src="https://github.com/silkdom/Tattoo-GAN/blob/master/img/Tattoo-GAN.gif?raw=true" alt="Tattoo-GAN"/>
</p>


## How it Works

The workflow for the generation of tattoo design ideas can be broken into 4 sections;

- Profile scraping
- Image scraping
- Model training
- Model testing

### Profile Scraping
[Notebook](https://github.com/silkdom/Tattoo-GAN/blob/master/Profile_Scraper.ipynb)

This section's objective is to compile a list of Instagram profiles of suspected tattoo artists (profiles likely to contain pictures of flash tattoo's). This logic behind constructing the workflow this way is because of the relitively low, variable, and unpublished request limit Instagram has on their servers. To bypass this, the idea is to maximize download efficiency (the chances that a image downloaded meets the requirements). Intuitively tattoo artists profiles maximize this efficiency. The first block of code first downloads all pictures in tagged #tattooflash over a certain time period. This is achieved using the Instaloader package with the metadata (date, likes, profile username) output as the the filename. 

```python
L = instaloader.Instaloader(save_metadata=False,download_comments=False,download_geotags=False,download_videos=False, filename_pattern="{date_utc}_UTC_likes_{likes}_profile_{profile}", post_metadata_txt_pattern="")

L.download_hashtag('tattooflash')
```

This block was then ran in the background (~4 days) to collect 1 month of posts (~100k images) in the #tattooflash tag. Using the file names, images are sorted by amount of 'likes' (so that artists generally appear first). For image filtering a defintion for a flashtattoo illustration has to be stated in computer readable form. This was decided to be a white background covering greater than 75% of the image. An example of an image meeting the requirements is as follows;

<p align="center">
  <img src="https://github.com/silkdom/Tattoo-GAN/blob/master/img/croc3.png?raw=true" alt="croc"/>
</p>

This filtering was achieved via the white() function, which specifies the proportion of white pixels in the image. 

```python
def white(img):
  im = Image.open(img)
  pixels = im.getdata()          
  white_thresh = 254
  nwhite = 0
  for pixel in pixels:
  
    if pixel[0] > white_thresh:
         nwhite += 1
  n = len(pixels)
  return(nwhite / float(n))
```

Using this function the vast majority of real (on body) tattoos and irrelavent images are deleted. The remainder of the notebook compiles a list of sorted profiles with atleast 1 image meeting the requirements. Of the ~100k imageset, 1479 unique profiles were satisfactory. 

### Image Scraping
[Notebook](https://github.com/silkdom/Tattoo-GAN/blob/master/Image_Scraper.ipynb)

Now a list of potential tattoo artsists is generated, Instaloader can then be used to download the contents of each profile. 

```python
for i in profiles:
    PROFILE = i
    posts = instaloader.Profile.from_username(L.context, PROFILE).get_posts()
    for post in posts:
        try:
            L.download_post(post, 'profiles_a')
        except:
            err += 1 # doing nothing on exception      
    cnt += move()
    prof += 1
    print('Total images: %f, profiles %f of %f' % (cnt, prof, len(profiles)))
```

Where the function move() mimics the profile scraper notebook and saves images in a profile that meet the requirements and deletes the rest. Sporadic unavoidable errors and rate limits were met so this script was run on three separate notebooks (to maximize uptime) for ~5 days, yielding ~50k satisfactory images. 

### Model training

First pass used the raw generated dataset with PyTorch's DCGAN implementation: [Repo](https://github.com/pytorch/examples/tree/master/dcgan). However, this is an older network architecture and provided less than adequate results. 

<p align="center">
  <img src="https://github.com/silkdom/Tattoo-GAN/blob/master/img/dcgan.png?raw=true" width="400" alt="dcgan"/>
</p>

Consequently, the decision was made to adopt a state of the art network architecture, thus landing on Nvidia's StyleGAN2: [Repo](https://github.com/NVlabs/stylegan2). Whilst best in class, this unfortunately required compute that my humble Macbook Pro could not provide, so I had to rent a VM (500GB / P5000 GPU) from [Paperspace](https://www.paperspace.com/). The training procedure applied is as follows;

The StyleGAN2 repo was first cloned and then the images were transferred from local to VM. This was done by; zipping the image folder, uploading to Google Drive, and a couple lines of terminal commands.

```.bash
gdown --id 'zipped file google drive id'
unzip 'zipped filename'
```

Next the imageset's size had to be standardized to the standard for StyleGAN2 is square 1024x1024. Dvschultz provides a great tool to do so: [Repo](https://github.com/dvschultz/dataset-tools). Instead of forcing the distortion of the non-square images, a white horizontal or veritcal border was simply added. 

```.bash
python dataset-tools.py --input_folder ~/stylegan2/raw_datasets/flashtattoo --output_folder ./output/datasets/ --process_type square --border_type solid --border_color 255,255,255
```

Whilst user friendly a few images were not standradized correctly, and the [size.py](https://github.com/silkdom/Tattoo-GAN/blob/master/size.py) script had to be created. This eliminated the outliers, leaving the imageset ready for training. 

The VM was expensive so in order to expadite the training process, trasnfer learning was used to kickstart the process. One of Nvidia's ffhq pretrained models were initialized as the starting .pkl file. Training was then commenced with mirror augmentation but without metric collection.

```.bash
python run_training.py --num-gpus=1 --data-dir=./datasets --config=config-f --dataset=sq-512-domo --mirror-augment=true --metrics=None
```

The initial results provide a fairly scary scene as the residual learnings of the pretrained (face) model still appear. 

<p align="center">
  <img src="https://github.com/silkdom/Tattoo-GAN/blob/master/img/fakes010016.jpg?raw=true" width="700" alt="fakes010016"/>
</p>

Thankfully the tormented faces fully dissapeared by the 60th iteration. The model was kept running on the VM for another 72 hours (411th iteration) until adequate results were produced. 

<p align="center">
  <img src="https://github.com/silkdom/Tattoo-GAN/blob/master/img/fakes010409.jpg?raw=true" width="700" alt="fakes010409"/>
</p>

It can be seen that there are a few similarities in the generated images, which can be unfortunately attributed to partial modal collapse. This has to be expected with such a diverse imageset, thus I am happy with the results! 

### Model Testing

The training process only produces 28 images per iteration (for progress tracking), and thus is not suitable for tattoo design generation. Instead a testing process using the now trained model can be performed. Thankfully the 'heavy-lifting' is not over and less compute is required. Infact for testing, Colab is more than adequate. Colab notebook soon to come. 

```.bash
!python run_generator.py generate-images --network=/content/network-snapshot-010409.pkl --seeds=1-1000 --truncation-psi=1.0
```

This command generates 1000 1024x1024 images, which can then be examined for tattoo inspiration. The following is my favourite, and will hopefully become permanent in the near future!

<p align="center">
  <img src="https://github.com/silkdom/Tattoo-GAN/blob/master/img/seed0179_2.png?raw=true" height="700" alt="Fav"/>
</p>

#### fix comments in Image scraping


