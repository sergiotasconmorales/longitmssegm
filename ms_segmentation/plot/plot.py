import numpy as np  
import matplotlib.pyplot as plt


def shim_slice(img, slice_index):
    fig, ax = plt.subplots()
    ax.imshow(img[:,:,slice_index], cmap = "gray")
    fig.set_facecolor("black")
    fig.tight_layout()
    ax.axis('off')
    plt.show()

def shim_overlay_slice(img, mask, slice_index, alpha = 0.7):
    masked = np.ma.masked_where(mask[:,:,slice_index] ==0, mask[:,:,slice_index])
    fig, ax = plt.subplots()
    ax.imshow(img[:,:, slice_index], 'gray', interpolation='none')
    ax.imshow(masked, 'jet', interpolation='none', alpha=alpha)
    fig.set_facecolor("black")
    fig.tight_layout()
    ax.axis('off')
    plt.show()

def shim(img, num_slices):
    if not num_slices <= img.shape[2]:
        raise ValueError("Number of slices to show must be smaller than total number of slices")
    else:

        if(num_slices<6):
            div=2
        else:
            div = 4
        med_val = int(np.ceil(num_slices/div))
        num_cols = med_val
        num_rows = div
        slices_to_show = np.linspace(0+5, img.shape[2]-5,  num_slices, dtype=int)
    fig, axes = plt.subplots(num_rows, num_cols)
    slice_index = 0
    for i in range(num_rows):
        for j in range(num_cols):
            if slice_index > num_slices-1:
                axes[i, j].imshow(np.zeros((img.shape[0], img.shape[1])), 'gray')
            else:
                axes[i, j].imshow(img[:,:,int(slices_to_show[slice_index])], 'gray')
                axes[i, j].axis('off')
                slice_index += 1


    fig.set_facecolor("black")
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()    



def shim_overlay(img, mask, num_slices, alpha=0.7):
    if not num_slices <= img.shape[2]:
        raise ValueError("Number of slices to show must be smaller than total number of slices")
    else:
        if(len(np.unique(mask)) < 2):
            print("Warning: Only one label in masks")
        if(num_slices<6):
            div=2
        else:
            div = 4
        med_val = int(np.ceil(num_slices/div))
        num_cols = med_val
        num_rows = div
        slices_to_show = np.linspace(0+5, img.shape[2]-5,  num_slices, dtype=int)
    fig, axes = plt.subplots(num_rows, num_cols)
    slice_index = 0
    for i in range(num_rows):
        for j in range(num_cols):
            if slice_index > num_slices-1:
                axes[i, j].imshow(np.zeros((img.shape[0], img.shape[1])), 'gray')
            else:
                masked = np.ma.masked_where( mask[:,:,int(slices_to_show[slice_index])] == 0, mask[:,:,int(slices_to_show[slice_index])] )
                axes[i, j].imshow(img[:,:,int(slices_to_show[slice_index])], 'gray')
                axes[i, j].imshow(masked, 'jet', interpolation='none', alpha=alpha)
                axes[i, j].axis('off')
                slice_index += 1


    fig.set_facecolor("black")
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()  



def plot_learning_curve(train_losses, val_losses, the_title="Figure", measure = "Loss", early_stopping = False, filename = "loss_plot.png", display = False):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(train_losses)+1),train_losses, label='Training '+ measure)
    plt.plot(range(1,len(val_losses)+1),val_losses,label='Validation ' + measure)

    # find position of lowest validation loss
    if early_stopping:
        minposs = val_losses.index(min(val_losses))+1 
        plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel(measure)
    plt.ylim(0, 0.5) # consistent scale
    plt.xlim(0, len(train_losses)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.ylim(( np.min([np.min(train_losses), np.min(val_losses)])-1,  np.max([np.max(train_losses), np.max(val_losses)]) +1))
    plt.title(the_title)
    if(display):
        plt.show()
    fig.savefig(filename, bbox_inches='tight')