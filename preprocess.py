
def reshape_patch(img_tensor, patch_size):
    img_tensor = img_tensor.permute([0,1,3,4,2])
    batch_size, seq_length, img_height, img_width, num_channels = img_tensor.shape
    img_tensor = img_tensor.reshape(batch_size, seq_length, img_height//patch_size, patch_size,
                           img_width//patch_size, patch_size, num_channels)
    img_tensor = img_tensor.permute([0, 1, 2, 4, 3, 5, 6])
    patch_tensor = img_tensor.reshape(batch_size, seq_length,
                                      img_height//patch_size, img_width//patch_size,
                                      patch_size*patch_size*num_channels)
    patch_tensor = patch_tensor.permute(0, 1, 4, 2, 3)
    return patch_tensor

def reshape_patch_back(patch_tensor, patch_size):
    patch_tensor = patch_tensor.permute([0, 1, 3, 4, 2])
    batch_size, seq_length, patch_height, patch_width, channels = patch_tensor.shape
    img_channels = channels // (patch_size*patch_size)
    patch_tensor = patch_tensor.reshape(batch_size, seq_length,
                                        patch_height, patch_width,
                                        patch_size, patch_size,
                                        img_channels)
    patch_tensor = patch_tensor.permute([0, 1, 2, 4, 3, 5, 6])
    img_tensor = patch_tensor.reshape(batch_size, seq_length,
                                      patch_height * patch_size, patch_width * patch_size,
                                      img_channels)
    img_tensor = img_tensor.permute(0, 1, 4, 2, 3)
    return img_tensor

