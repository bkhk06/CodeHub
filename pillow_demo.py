from PIL import Image
im=Image.open("lenna.png")

print(im.format,im.size,im.mode)

from PIL import ImageFilter

outF = im.filter(ImageFilter.DETAIL)
conF = im.filter(ImageFilter.CONTOUR)
edgeF = im.filter(ImageFilter.FIND_EDGES)
im.show()
outF.show()
conF.show()
edgeF.show()

from PIL import ImageEnhance

imgE = Image.open("lenna.png")
imgEH = ImageEnhance.Contrast(imgE)
imgE.show()
imgEH.enhance(1.3).show("30% more contrast")
imgEH.enhance(1.8).show("80% more contrast")
