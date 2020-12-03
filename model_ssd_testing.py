import d2l
from mxnet import autograd, contrib, gluon, image, init, nd
from mxnet.gluon import loss as gloss, nn
import time


def cls_predictor(num_anchors, num_classes):
    """Predictor layer
    For each anchor(box), we need to generate num_clases+1 predictions

    Args:
        num_anchors (int): Num boxes
        num_classes (int): Num classes

    Returns:
        [type]: Input.shape=Output.shape
    """
    return Conv2D(num_anchors*(num_classes+1), kernel_size=3, padding=1)

def forward(x, block):
    """Sanity check

    Args:
        x ([type]): [description]
        block ([type]): [description]

    Returns:
        [type]: [description]
    """
    block.initialize()
    return block(x)
"""(2, 8, 20, 20)-->(batch_size, num_channels, feature_map_width, feature_map_height)
"""
Y1=forward(nd.zeros((2, 8, 20, 20)),cls_predictor(5, 10))#Output--> (2L, 55L, 20L, 20L); 55=num_anchors*(num_clases+1)
Y1=forward(nd.zeros((2, 16, 10, 10)),cls_predictor(3, 10))#Output--> (2L, 33L, 10L, 10L); 33=num_anchors*(num_clases+1)


#Concatenating predictions for multiple scales

def flatten_pred(pred):
    return pred.transpose((0, 2, 3, 1)).flatten()

def concat_preds(pred):
    return nd.concat(*[flatten_pred(p for p in preds)], dim=1)

concat_preds([Y1, Y2]).shape #(2L, 25300L)

#Bounding box prediction layer

def bbox_predictor(num_anchors):
    return Conv2D(num_anchors*4, kernel_size=3, padding=1) #same output shape as input shape

#Height and Widht Downsample block

def down_sample_blk(num_channels):
    blk=nn.Sequential()
    for _ in range(2):
        blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1),
                nn.BatchNorm(in_channels=num_channels),
                nn.Activation('relu'))
    blk.add(nn.MaxPool2D(2)) #reduce height and width by half
    return blk

forward(nd.zeros((2, 3, 20, 20)), down_sample_blk(10)).shape #(2L, 10L, 10L, 10L)

#base network block, we can use a pretrainned net--> resnet or vgg16

def base_net():
    blk = nn.Sequential()
    for num_filters in [16,32,64]: #3 down_sample channels as the base net
        blk.add(down_sample_blk(num_filters))
    return blk
#input=(2=batch, 3=RGB, input_height=256, input_height=256)
forward(nd.zeros((2, 3, 256, 256)), base_net()).shape #(2L, 64L, 32L, 32L)

def get_blk(i):
    """ 
    0.- base_net
    1, 2, 3.- down_sample_blk
    4.- GlobalMaxPool2D

    Args:
        i ([type]): [description]

    Returns:
        [type]: [description]
    """
    if i==0:
        blk=base_net()
    elif i==4:
        blk= nn.GlobalMaxPool2D()
    else:
        blk = down_sample_blk(128)
    return blk


def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    """[summary]

    Args:
        X ([type]): [description]
        blk ([type]): Model
        size ([type]): Anchorbox size
        ratio ([type]): Ratio for anchor box
        cls_predictor ([type]): Anchor box predictor
        bbox_predictor ([type]): [description]

    Returns:
        Y (featuremap): Input for the next block
        cls_preds, and bbox_preds for every pixel we have
    """
    Y=blk(X) #Y is a feature map
    anchors=contrib.ndarray.MultiBoxPrior(Y, sizes=size, ratios=ratio)#Generate anchors for this block
    cls_preds=cls_predictor(Y) #
    bbox_preds=bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)

sizes=[[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88,0.961]] #linear scale, semi-random, 2ºvalue is sqr(1º*3º)
ratios=[[1, 2, 0.5]]* 5
num_anchors=len(sizes[0]) + len(ratios[0]) -1 #for each pixel it generates 4 anchor boxes


class TinySSD(nn.Block):
    """Will give you:
            -anchor boxes we have
            -class prediction
            -bbox prediction

    Args:
        nn ([type]): [description]
    """
    def __init__(self, num_classes, **kwargs):
        super(TinySDD, self).__init__(**kwargs)
        self.num_classes=num_classes
        for i in range(5):
            #setattr == self.blk_i=get_blk(i), para no ensuciar
            setattr(self, 'blk_%d' % i, get_blk(i))
            setattr(self, 'cls_%d' % i,cls_predictor(num_anchors, num_classes))
            setattr(self, 'bbox_%d' % i,bbox_predictor(num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds= [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(X, getattr(self, 'blk_ %d' % i), sizes[i], ratios[i], 
                                                                        getattr(self, 'cls_%d' % i), getattr(self, 'bbox_%d' % i))
        return (nd.concat(*anchors, dim=1), 
                concat_preds(cls_preds).reshape(
                    (0,-1, self.num_classes+1)),
                concat_preds(bbox_preds))


#sanity check. A total of (32² + 16² +8²+4²+1)*4= 5444 anchor boxes are generated for each image at the five scales


net=TinySSD(num_classes=1)
net.initialize()
X=nd.zeros((32,3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors: ', anchors.shape) #(1L, 5444L, 4L)
print('output class preds: ', cls_preds.shape) #(32L, 5444L, 2L)
print('output bbox preds: ', bbox_preds.shape) #(32L, 21776L)




#Training

batch_size=32
train_iter, _ = d2l.load_data_pikachu(batch_size)
ctx, net = d2l.try_gpu(), TinySSD(num_classes=1)
net.initialize(init=init.Xavier(), ctx=ctx)
trainer= gluon.Trainer(net.collect_params(),'sgd', {'learning_rate':0.2, 'wd':5e-4})


#Define Losses

cls_loss = gloss.SoftmaxCrossEntropyLoss()
bbox_loss= gloss.L1Loss()

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    cls=cls_loss(cls_preds, cls_labels)
    bbox=bbox_loss(bbox_preds * bbox_masks, bbox_labels*bbox_masks)
    return cls+bbox



#evaluation metrics
def cls_eval(cls_preds, cls_labels):
    return (cls_preds.argmax(axis=-1) == cls_labels).sum().asscalar()

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return ((bbox_labels-bbox_preds)*bbox_masks).abs().sum().asscalar()


#Training the model

for epoch in range(20):
    acc_sum, mae_sum, n, m=0.0,0.0,0,0
    train_iter.reset()
    start = time.time()
    for batch in train_iter:
        X = batch.data[0].as_in_context(ctx)
        X = batch.label[0].as_in_context(ctx)
        with autograd.record():
            #generate multiscale anchor boxes and predict the category and offset of each
            anchors, cls_preds, bbox_preds = net(X)
            #label the category and offset of each anchor box
            bbox_labels, bbox_masks, cls_labels = contrib.nd.MultiBoxTarget(
                anchors, Y, cls_preds.transpose((0, 2, 1))
            )
            #calculate the loss function using the predicted and labeled category and offset
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
        l.backward()
        trainer.step(batch_size)
        acc_sum+=cls_eval(cls_preds, cls_labels)
        n+=cls_labels.size
        mae_sum+=bbox_eval(bbox_preds, bbox_labels, bbox_masks)
        m+=bbox_labels.size
    if (epoch +1)%5==0:
        print('epoch %2d, class err %.2e, bbox mae %.2e, time %.1f sec' % (
                epoch+1, 1-acc_sum/n, mae_sum/m, time.time()-start))


#Predictions once the network is trained
def predict(X):
    anchors, cls_preds, bbox_preds = net(X.as_in_context(ctx))
    cls_probs = cls_preds.softmax().transpose((0, 2, 1))
    output = contrib.nd.MultiBoxDetection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0].asscalar() !=-1]
    return output[0, idx]


image = image.imread('') ##read the image
feature= image.imresize(img, 256, 256).astype('float32')
X= feature.transpose((2, 0, 1)).expand_dims(axis=0)

output = predict(X)



#visualize the results

def display(img, output, threshold):
    fig = d2l.plt.imshow(img.asnumpy())
    for row in output:
        score = row[1].asscalar()
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox =[row[2:6]*nd.array((w, h, w, h),ctx=row.context)]
        d2l.show_boxes(fig.axes, bbox, '%.2f' % score, 'w')


display(img, output, threshold=0.3)



