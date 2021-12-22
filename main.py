import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
matplotlib.rcParams['animation.embed_limit'] = 2**128
from IPython.display import HTML

from LLD_Dataset import LLD_Dataset

# Model Parameters
batch_size = 16
shuffle = True
cuda_enabled = True
learning_rate = 0.0002
num_epochs = 1


# Data Parameters
image_dim = 128
z_size = 100
y_size_padded = 50
total_yz_size = z_size + y_size_padded



# Data loader
image_transform = transforms.Compose([transforms.Resize((image_dim, image_dim))])
data_loader = DataLoader(dataset=LLD_Dataset(transform=image_transform, text_buffer_to=y_size_padded), batch_size=batch_size, shuffle=shuffle, drop_last=True)



def zy_builder(label_y, device):
    z_samples = np.random.normal(0, 1, size=(batch_size, z_size))
    z_samples = torch.FloatTensor(z_samples).to(device)
    print(z_samples.shape)
    print(label_y.shape)
    zy = torch.cat([z_samples, label_y], 1)
    zy = torch.reshape(zy, (batch_size, total_yz_size, 1, 1))
    return zy

def zy_builder_no_batch(label_y, device):
    z_samples = np.random.normal(0, 1, size=(1, z_size))
    z_samples = torch.FloatTensor(z_samples).to(device)
    print(z_samples.shape)
    print(label_y.shape)
    zy = torch.cat([z_samples, label_y], 1)
    zy = torch.reshape(zy, (1, total_yz_size, 1, 1))
    return zy


def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(model.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(model.bias.data, 0)




# Generator model
class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(total_yz_size, 2048, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(2048, momentum=0.9),
            torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
            torch.nn.ConvTranspose2d(2048, 1024, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(1024, momentum=0.9),
            torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
            torch.nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(512, momentum=0.9),
            torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
            torch.nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(256, momentum=0.9),
            torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
            torch.nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(128, momentum=0.9),
            torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
            torch.nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            torch.nn.Tanh()
        )

    def forward(self, input):
        return self.layers(input)


# Discriminator Model
class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_flattener = torch.nn.Sequential(
            torch.nn.Linear(y_size_padded, image_dim * image_dim)
        )
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(4, 128, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(1024, 2048, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(2048),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(2048, 1, 4, 1, 0, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, input, label):
        flat_label = torch.reshape(self.label_flattener(label), (batch_size, 1, image_dim, image_dim))
        combine = torch.cat((input, flat_label), 1)
        return self.layers(combine)


if cuda_enabled is True:
    if torch.cuda.is_available():
        print(f"CUDA enabled and available. Engaging...")
        device = 'cuda:0'
    else:
        raise OSError(f"CUDA enabled but not available. Exiting")
else:
    device = 'cpu'


gennie = Generator()
gennie.load_state_dict(torch.load("Modelv1"))
gennie.to("cuda:0")

text_label = torch.FloatTensor(np.array([99]))
label = torch.cat((text_label, torch.zeros(50-text_label.shape[0])))
label = torch.reshape(label, [1, 50])
zy = zy_builder_no_batch(label.to("cuda:0"), "cuda:0")
gennie.eval()
genned = gennie(zy)
plt.imshow(np.array(genned[0].detach().cpu().permute(1, 2, 0)) / 2 + 0.5)
plt.show()


net_G = Generator().to(device)
net_G.apply(weights_init)
print(net_G)

net_D = Discriminator().to(device)
net_D.apply(weights_init)
print(net_D)

criterion_D = torch.nn.BCELoss()
criterion_G = torch.nn.MSELoss()

real_label = 0
fake_label = 1

optimizerD = torch.optim.Adam(net_D.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(net_D.parameters(), lr=learning_rate, betas=(0.5, 0.999))

img_list = []
G_losses = []
D_losses = []
iters = 0

fixed_zy = None


print("Starting Training...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, package in enumerate(data_loader, 0):
        data, name_label = package[0], package[1]
        data = data.to(device)
        name_label = name_label.to(device)

        # (1) Update Discriminator #################################################################################
        ### Train discriminator on real data
        net_D.zero_grad()
        label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = net_D(data, name_label).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion_D(output, label)
        # Calculate gradients for D in backward pass
        #errD_real.backward()
        D_x = output.mean().item()

        ### Train discriminator on generator data
        net_G.zero_grad()
        label = torch.full((batch_size,), fake_label, dtype=torch.float, device=device)
        noise_and_label = zy_builder(name_label, device)
        if fixed_zy is None:
            fixed_zy = noise_and_label.detach()
        fake_generated = net_G(noise_and_label)
        output = net_D(fake_generated.detach(), name_label).view(-1)
        errD = criterion_D(output, label)
        errD_total = errD_real + errD_real
        errD_total.backward()
        #errD.backward()
        D_G_zy1 = output.mean().item()
        optimizerD.step()


        # (2) Update Generator #################################################################################
        net_G.zero_grad()
        label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
        plt.imshow(np.array(fake_generated[0].detach().cpu().permute(1, 2, 0)) / 2 + 0.5)
        plt.show()
        output = net_D(fake_generated, name_label).view(-1)
        errG = criterion_D(output, label)
        errG.backward()
        D_G_zy2 = output.mean().item()
        optimizerG.step()

        if i % 10 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(data_loader),
                     errD.item(), errG.item(), D_x, D_G_zy1, D_G_zy2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 10 == 0) or ((epoch == num_epochs-1) and (i == len(data_loader)-1)):
            with torch.no_grad():
                fake_generated = net_G(fixed_zy).detach().cpu()
                plt.imshow(np.array(fake_generated[0].detach().cpu().permute(1, 2, 0)) / 2 + 0.5)
                plt.show()
            img_list.append(vutils.make_grid(fake_generated, padding=2, normalize=True))

        iters += 1

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())

torch.save(net_G.state_dict(), "Modelv1")

