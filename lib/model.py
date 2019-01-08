from keras.layers import Input, Dense, Conv2D, Add, Dot, Conv2DTranspose, Activation, Reshape,BatchNormalization,UpSampling2D,AveragePooling2D, GlobalAveragePooling2D, LeakyReLU, Reshape, Flatten
from keras.models import Model, Sequential
import keras.backend as K
from keras.utils import plot_model
from SpectralNormalizationKeras import DenseSN, ConvSN2D
from keras.layers.pooling import _GlobalPooling2D
from keras.layers.merge import concatenate

class GlobalSumPooling2D(_GlobalPooling2D):
    """Global sum pooling operation for spatial data.
    # Arguments
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    # Input shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, rows, cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, rows, cols)`
    # Output shape
        2D tensor with shape:
        `(batch_size, channels)`
    """

    def call(self, inputs):
        if self.data_format == 'channels_last':
            return K.sum(inputs, axis=[1, 2])
        else:
            return K.sum(inputs, axis=[2, 3])


def ResBlock(input_shape, sampling=None, trainable_sortcut=True, 
             spectral_normalization=False, batch_normalization=True,
             bn_momentum=0.9, bn_epsilon=0.00002,
             channels=256, k_size=3, summary=False,
             plot=False, name=None):
    '''
    ResBlock(input_shape, sampling=None, trainable_sortcut=True, 
             spectral_normalization=False, batch_normalization=True,
             bn_momentum=0.9, bn_epsilon=0.00002,
             channels=256, k_size=3, summary=False,
             plot=False, plot_name='res_block.png')""
             
    Build ResBlock as keras Model
    sampleing = 'up' for upsampling
                'down' for downsampling(AveragePooling)
                None for none
    
    '''
    #input_shape = input_layer.sahpe.as_list()
    
    res_block_input = Input(shape=input_shape)
    
    if batch_normalization:
        res_block_1 = BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon)(res_block_input)
    else:
        res_block_1 = res_block_input
        
    res_block_1     = Activation('relu')(res_block_1)
    
    if spectral_normalization:
        res_block_1     = ConvSN2D(channels, k_size , strides=1, padding='same',kernel_initializer='glorot_uniform')(res_block_1)
    else:
        res_block_1     = Conv2D(channels, k_size , strides=1, padding='same',kernel_initializer='glorot_uniform')(res_block_1)
    
    if sampling=='up':
        res_block_1     = UpSampling2D()(res_block_1)
    else:
        pass
    
    if batch_normalization:
        res_block_2     = BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon)(res_block_1)
    else:
        res_block_2     = res_block_1
    res_block_2     = Activation('relu')(res_block_2)
    
    if spectral_normalization:
        res_block_2     = ConvSN2D(channels, k_size , strides=1, padding='same',kernel_initializer='glorot_uniform')(res_block_2)
    else:
        res_block_2     = Conv2D(channels, k_size , strides=1, padding='same',kernel_initializer='glorot_uniform')(res_block_2)
    
    if sampling=='down':
        res_block_2 = AveragePooling2D()(res_block_2)
    else:
        pass
    
    if trainable_sortcut:
        if spectral_normalization:
            short_cut = ConvSN2D(channels, 1 , strides=1, padding='same',kernel_initializer='glorot_uniform')(res_block_input)
        else:
            short_cut = Conv2D(channels, 1 , strides=1, padding='same',kernel_initializer='glorot_uniform')(res_block_input)
    else:
        short_cut = res_block_input
        
    if sampling=='up':
        short_cut       = UpSampling2D()(short_cut)
    elif sampling=='down':
        short_cut       = AveragePooling2D()(short_cut)
    elif sampling=='None':
        pass

    res_block_add   = Add()([short_cut, res_block_2])
    
    res_block = Model(res_block_input, res_block_add, name=name)
    
    if plot:
        plot_model(res_block, name+'.png', show_layer_names=False)
    if summary:
        print(name)
        res_block.summary()
    
    return res_block


def generator(input_shape,
              noise_code,
              pc_code,
              elev_code,
              azim_code,
              color=False,
              spectral_normalization=False,
              bn_momentum=0.9,
              bn_epsilon=0.00002):

    if spectral_normalization:
        conv = ConvSN2D
        dense = DenseSN
    else:
        conv = Conv2D
        dense = Dense

    # inputs = Input(input_shape, name='noise')
    inputs = [noise_code, pc_code, elev_code, azim_code]
    x = concatenate(inputs, axis=1)

    # magic = [(8, 8), (8, 4), (4, 4), (4, 2), (2, 1)]
    in_channels = 1024
    d = input_shape[0] // 32
    x = dense(d * d * in_channels,
              kernel_initializer='glorot_uniform')(x)
    x = Reshape((d, d, in_channels))(x)

    name = 'gen_resblock_0'
    shape = (d, d, in_channels)
    out_channels = 1024
    x = ResBlock(input_shape=shape,
                 channels=out_channels,
                 sampling='up',
                 spectral_normalization=spectral_normalization,
                 bn_epsilon=bn_epsilon,
                 bn_momentum=bn_momentum,
                 name=name)(x)


    name = 'gen_resblock_1'
    d = input_shape[0] // 16
    shape = (d, d, in_channels)
    out_channels = 512
    x = ResBlock(input_shape=shape,
                 channels=out_channels,
                 sampling='up',
                 spectral_normalization=spectral_normalization,
                 bn_epsilon=bn_epsilon,
                 bn_momentum=bn_momentum,
                 name=name)(x)

    name = 'gen_resblock_2'
    d = input_shape[0] // 8
    in_channels = 512
    shape = (d, d, in_channels)
    out_channels = 256
    x = ResBlock(input_shape=shape,
                 channels=out_channels,
                 sampling='up',
                 spectral_normalization=spectral_normalization,
                 bn_epsilon=bn_epsilon,
                 bn_momentum=bn_momentum,
                 name=name)(x)

    name = 'gen_resblock_3'
    d = input_shape[0] // 4
    in_channels = 256
    shape = (d, d, in_channels)
    out_channels = 128
    x = ResBlock(input_shape=shape,
                 channels=out_channels,
                 sampling='up',
                 spectral_normalization=spectral_normalization,
                 bn_epsilon=bn_epsilon,
                 bn_momentum=bn_momentum,
                 name=name)(x)

    name = 'gen_resblock_4'
    d = input_shape[0] // 2
    in_channels = 128
    shape = (d, d, in_channels)
    out_channels = 64
    x = ResBlock(input_shape=shape,
                 channels=out_channels,
                 sampling='up',
                 spectral_normalization=spectral_normalization,
                 bn_epsilon=bn_epsilon,
                 bn_momentum=bn_momentum,
                 name=name)(x)

    # for i in range(5):
    #    k = 2 ** i
    #    name = "gen_resblock_" + str(i)
    #    x = ResBlock(input_shape=(d * k, d * k, channels),
    #                 channels=(channels // 2),
    #                 sampling='up',
    #                 bn_epsilon=bn_epsilon,
    #                 bn_momentum=bn_momentum,
    #                 name=name)(x)
    #    channels //= 2

    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(x)
    x = Activation('relu')(x)

    filters = 1
    if color:
        filters = 3

    outputs = conv(filters,
                   kernel_size=3,
                   strides=1,
                   padding='same',
                   activation='tanh',
                   name='fake_image')(x)
    return Model(inputs, outputs, name="generator")
    

def discriminator(input_shape,
                  pc_code_dim=0,
                  spectral_normalization=True,
                  batch_normalization=False,
                  bn_momentum=0.9,
                  bn_epsilon=0.00002):

    inputs = Input(shape=input_shape, name='image_input')
    channels = 64
    x = ResBlock(input_shape=input_shape,
                 channels=channels,
                 sampling='down',
                 batch_normalization=batch_normalization,
                 spectral_normalization=spectral_normalization,
                 name='dis_resblock_0')(inputs)

    d = input_shape[0] // 2
    shape = (d, d, channels)
    name = 'dis_resblock_1'
    x = ResBlock(input_shape=shape,
                 channels=(channels * 2),
                 sampling='down',
                 batch_normalization=batch_normalization,
                 spectral_normalization=spectral_normalization,
                 name=name)(x)
    channels *= 2

    d = input_shape[0] // 4
    shape = (d, d, channels)
    name = 'dis_resblock_2'
    x = ResBlock(input_shape=shape,
                 channels=(channels * 2),
                 sampling='down',
                 batch_normalization=batch_normalization,
                 spectral_normalization=spectral_normalization,
                 name=name)(x)
    channels *= 2

    d = input_shape[0] // 8
    shape = (d, d, channels)
    name = 'dis_resblock_3'
    x = ResBlock(input_shape=shape,
                 channels=(channels * 2),
                 sampling='down',
                 batch_normalization=batch_normalization,
                 spectral_normalization=spectral_normalization,
                 name=name)(x)
    channels *= 2
    aux_layer_pc = Flatten(name='aux_layer_pc')(x)

    d = input_shape[0] // 16
    shape = (d, d, channels)
    name = 'dis_resblock_4'
    x = ResBlock(input_shape=shape,
                 channels=(channels * 2),
                 sampling='down',
                 batch_normalization=batch_normalization,
                 spectral_normalization=spectral_normalization,
                 name=name)(x)
    channels *= 2

    d = input_shape[0] // 32
    shape = (d, d, channels)
    name = 'dis_resblock_5'
    x = ResBlock(input_shape=shape,
                 channels=channels,
                 sampling=None,
                 batch_normalization=batch_normalization,
                 spectral_normalization=spectral_normalization,
                 name=name)(x)


    x = Activation('relu')(x)
    x = GlobalSumPooling2D()(x)
    preal = DenseSN(1, activation='sigmoid', kernel_initializer='glorot_uniform', name='real_fake')(x)

    aux_layer_pc = DenseSN(256, activation='relu')(aux_layer_pc)
    reco_pc_code = DenseSN(1024, activation='relu')(aux_layer_pc)
    reco_pc_code = DenseSN(1024, activation='relu')(reco_pc_code)
    reco_pc_code = DenseSN(1024, activation='relu')(reco_pc_code)
    reco_pc_code = DenseSN(pc_code_dim, activation='linear', name='reco_pc_code')(reco_pc_code)

    reco_azim_code = DenseSN(1, activation='sigmoid', kernel_initializer='glorot_uniform', name='reco_azim_code')(x)

    reco_elev_code = DenseSN(1, activation='sigmoid', kernel_initializer='glorot_uniform', name='reco_elev_code')(x)

    outputs = [preal, reco_pc_code, reco_elev_code, reco_azim_code]

    return Model(inputs, outputs, name='discriminator')

def discriminator_aux(input_shape,
                      pc_code_dim=0,
                      bn_momentum=0.9,
                      bn_epsilon=0.00002):

    inputs = Input(shape=input_shape, name='image_input')
    channels = 64
    x = ResBlock(input_shape=input_shape,
                 channels=channels,
                 sampling='down',
                 batch_normalization=False,
                 name='dis_resblock_0')(inputs)

    d = input_shape[0] // 2
    shape = (d, d, channels)
    name = 'dis_resblock_1'
    x = ResBlock(input_shape=shape,
                 channels=(channels * 2),
                 sampling='down',
                 batch_normalization=False,
                 name=name)(x)
    channels *= 2

    d = input_shape[0] // 4
    shape = (d, d, channels)
    name = 'dis_resblock_2'
    x = ResBlock(input_shape=shape,
                 channels=(channels * 2),
                 sampling='down',
                 batch_normalization=False,
                 name=name)(x)
    channels *= 2

    d = input_shape[0] // 8
    shape = (d, d, channels)
    name = 'dis_resblock_3'
    x = ResBlock(input_shape=shape,
                 channels=(channels * 2),
                 sampling='down',
                 batch_normalization=False,
                 name=name)(x)
    channels *= 2
    aux_layer = Flatten(name='aux_layer')(x)

    d = input_shape[0] // 16
    shape = (d, d, channels)
    name = 'dis_resblock_4'
    x = ResBlock(input_shape=shape,
                 channels=(channels * 2),
                 sampling='down',
                 batch_normalization=False,
                 name=name)(x)
    channels *= 2

    d = input_shape[0] // 32
    shape = (d, d, channels)
    name = 'dis_resblock_5'
    x = ResBlock(input_shape=shape,
                 channels=channels,
                 sampling=None,
                 batch_normalization=False,
                 name=name)(x)


    x = Activation('relu')(x)
    x = GlobalSumPooling2D()(x)
    preal = DenseSN(1, activation='sigmoid', kernel_initializer='glorot_uniform', name='real_fake')(x)

    aux_layer = DenseSN(256, activation='relu')(aux_layer)
    reco_pc_code = DenseSN(1024, activation='relu')(aux_layer)
    reco_pc_code = DenseSN(1024, activation='relu')(reco_pc_code)
    reco_pc_code = DenseSN(1024, activation='relu')(reco_pc_code)
    reco_pc_code = DenseSN(pc_code_dim, activation='linear', name='reco_pc_code')(reco_pc_code)

    reco_elev_code = DenseSN(128, activation='relu')(aux_layer)
    reco_elev_code = DenseSN(128, activation='relu')(reco_elev_code)
    reco_elev_code = DenseSN(128, activation='relu')(reco_elev_code)
    reco_elev_code = DenseSN(1, activation='sigmoid', kernel_initializer='glorot_uniform', name='reco_elev_code')(reco_elev_code)

    reco_azim_code = DenseSN(128, activation='relu')(aux_layer)
    reco_azim_code = DenseSN(128, activation='relu')(reco_azim_code)
    reco_azim_code = DenseSN(128, activation='relu')(reco_azim_code)
    reco_azim_code = DenseSN(1, activation='sigmoid', kernel_initializer='glorot_uniform', name='reco_azim_code')(reco_azim_code)

    outputs = [preal, reco_pc_code, reco_elev_code, reco_azim_code]

    return Model(inputs, outputs, name='discriminator')

if __name__ == '__main__':
    from keras.utils import plot_model
    
    input_shape = (128,)
    view_code = Input(shape=(2,), name='view_code')
    pc_code = Input(shape=(32,), name='pc_code')
    model = generator(input_shape, pc_code=pc_code, view_code=view_code)
    model.summary()
    plot_model(model, show_shapes=True, to_file='saved_models/generator.png')

    input_shape = (128, 128, 1)
    model = discriminator(input_shape, pc_code_dim=32, view_code_dim=2)
    model.summary()
    plot_model(model, show_shapes=True, to_file='saved_models/discriminator.png')
