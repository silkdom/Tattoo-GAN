# Tattoo-GAN

TBC Soon! 6/22/2020

<p align="center">
  <img src="https://github.com/silkdom/Tattoo-GAN/blob/master/img/Tattoo-GAN.gif?raw=true" alt="Tattoo-GAN"/>
</p>

Using GANs to create tattoo ideas

## Overview

## How it Works

The workflow for the generation of tattoo design ideas can be broken into 3 sections;

- Profile scraping
- Image scraping
- Tattoo design generation

#### Profile Scraping
[Notebook](https://github.com/silkdom/Tattoo-GAN/blob/master/Profile_Scraper.ipynb)

This section's objective is to compile a list of Instagram profiles that are likely to contain pictures of flash tattoo's. This logic behind constructing the workflow this way is because of the relitively low, variable, and unpublished request limit Instagram has on their servers. To bypass this, the first block of code first downloads all pictures in tagged #tattooflash over a certain time period. This is achieved using the Instaloader package with the metadata (date, likes, profile username) output as the the filename. 

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

#### Image Scraping
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

#### Tattoo design generation

First pass used the raw generated dataset with PyTorch's DCGAN implementation: [Repo](https://github.com/pytorch/examples/tree/master/dcgan). However, this is an older network architecture and provided less than adequate results. 

<p align="center">
  <img src="https://github.com/silkdom/Tattoo-GAN/blob/master/img/dcgan.png?raw=true" width="400" alt="dcgan"/>
</p>

Consequently, the decision was made to adopt a state of the art network architecture, thus landing on Nvidia's StyleGAN2: [Repo](https://github.com/NVlabs/stylegan2). Whilst best in class, this unfortunately required compute that my humble Macbook Pro could not provide, so I had to rent a VM (500GB / P5000 GPU) from [Paperspace](https://www.paperspace.com/). The testing procedure applies is as follows;





```.bash
python dataset-tools.py --input_folder ~/stylegan2/raw_datasets/agg --output_folder ./output/agg/ --process_type square --border_type solid --border_color 255,255,255
```

## Data

## fix comments in Image scraping


