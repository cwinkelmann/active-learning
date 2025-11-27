import qrcode
import matplotlib.pyplot as plt

# URL to encode
url = "https://figshare.com/account/articles/30148708?file=58038142"

# Generate QR code
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=40,
    border=4,
)
qr.add_data(url)
qr.make(fit=True)

# Create an image from the QR Code instance
img = qr.make_image(fill_color="black", back_color="white")

# Display the QR code
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()
, dpi=(1200, 1200)