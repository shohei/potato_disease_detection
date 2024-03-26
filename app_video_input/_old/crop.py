from PIL import Image
import matplotlib.pyplot as plt

im = Image.open('leaf.png')
width, height = im.size   # Get dimensions
new_width = 256
new_height = 256
left = (width - new_width)/2
top = (height - new_height)/2
right = (width + new_width)/2
bottom = (height + new_height)/2
im = im.crop((left, top, right, bottom))

#new_img = img.resize((256, 256))

plt.imshow(im)
plt.show()



