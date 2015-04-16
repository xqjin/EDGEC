#!/usr/bin/python
"""
Modify the code: Convolutional Neural Networks for Sentence Classification 

	for GEC task;

Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf

Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
- https://github.com/yoonkim/CNN_sentence


"""
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys
from Key2Value import PrepKV as PKV
from Key2Value import ArtKV as AKV
from Key2Value import NnKV as NKV
import configure
from ProcessBar import progress

warnings.filterwarnings("ignore")   

#different non-linearities
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)
       
def train_conv_net(datasets,
                   U,
                   img_w=50, 
                   filter_hs=[3,4,5],
                   hidden_units=[100,2], 
                   dropout_rate=[0.5],
                   shuffle_batch=True,
                   n_epochs=25, 
                   batch_size=50, 
                   lr_decay = 0.95,
                   conv_non_linear="relu",
                   activations=[Iden],
                   sqr_norm_lim=9,
                   non_static=True):
    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes    
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """    
    rng = np.random.RandomState(3435)
    img_h = len(datasets[0][0])-1 # each instance in train and test is a list of ids; id can be mapped to a vector! 
    filter_w = img_w    
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
    parameters = [("image shape",img_h,img_w),("filter shape",filter_shapes), ("hidden_units",hidden_units),
                  ("dropout", dropout_rate), ("batch_size",batch_size),("non_static", non_static),
                    ("learn_decay",lr_decay), ("conv_non_linear", conv_non_linear), ("non_static", non_static)
                    ,("sqr_norm_lim",sqr_norm_lim),("shuffle_batch",shuffle_batch)]
    print parameters    
    
    #define model architecture
    index = T.lscalar()
    x = T.matrix('x')   
    y = T.ivector('y')
    Words = theano.shared(value = U, name = "Words")
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))])
    """
    x.flatten(): turn the matrix to one line 
    T.cast()   : ture the data type
    Words[np.array([...index...])]: read the Word[...index...] line;
    	Example:
    		words = [
    				[1,2,3],
    				[2,3,4],
    				[3,4,5]]
    		index =np.array([0,1,1,1])
    		words[index] = [[1,2,3],[2,3,4],[2,3,4],[2,3,4] ]
    reshape() : ture the shape of the object;

    At last, layer0_input[0] = [[vec],[vec],[vec]]  
    Here we get the input method for our GEC
    """
    layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],Words.shape[1]))                                  
    conv_layers = []
    layer1_inputs = []
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        # call the function in the conv_net_classes.py 
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)  # flatten(2) :  turn the object to one line in order of columon!
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    layer1_input = T.concatenate(layer1_inputs,1)
    hidden_units[0] = feature_maps*len(filter_hs)    
    classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)
    
    #define parameters of the model and update functions using adadelta
    params = classifier.params     
    for conv_layer in conv_layers:
        params += conv_layer.params
    if non_static:
        #if word vectors are allowed to change, add them as model parameters
        params += [Words]
    cost = classifier.negative_log_likelihood(y) 
    dropout_cost = classifier.dropout_negative_log_likelihood(y)           
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)
    
    #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate 
    #extra data (at random)
    np.random.seed(3435)
    #if the train data is not the multiple of mini batches , replicate the data to satify the condition
    if datasets[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        train_set = np.random.permutation(datasets[0])   
        extra_data = train_set[:extra_data_num]
        new_data=np.append(datasets[0],extra_data,axis=0)
    else:
        new_data = datasets[0]
    new_data = np.random.permutation(new_data)  #permutate the list: balance the data of different tags;
    n_batches = new_data.shape[0]/batch_size
    n_train_batches = int(np.round(n_batches*0.9))
    #divide train set into train/val sets 
    test_set_x = datasets[1][:,:img_h] 
    test_set_y = np.asarray(datasets[1][:,-1],"int32")
    train_set = new_data[:n_train_batches*batch_size,:]
    val_set = new_data[n_train_batches*batch_size:,:]     
    train_set_x, train_set_y = shared_dataset((train_set[:,:img_h],train_set[:,-1]))
    val_set_x, val_set_y = shared_dataset((val_set[:,:img_h],val_set[:,-1]))
    #n_val_batches: number of the validate batches
    n_val_batches = n_batches - n_train_batches

    val_model = theano.function([index], classifier.errors(y),
         givens={
            x: val_set_x[index * batch_size: (index + 1) * batch_size],
            y: val_set_y[index * batch_size: (index + 1) * batch_size]})
            
    #compile theano functions to get train/val/test errors
    test_model = theano.function([index], classifier.errors(y),
             givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]})               

    train_model = theano.function([index], cost, updates=grad_updates,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
            y: train_set_y[index*batch_size:(index+1)*batch_size]})     
	

    #####IMPORTMANT#######
    # This block is for test the model's error!
    test_pred_layers = []
    test_size = test_set_x.shape[0]
    test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))
    for conv_layer in conv_layers:
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))
    test_layer1_input = T.concatenate(test_pred_layers, 1)
    test_y_pred ,test_y_prob = classifier.predict_tp(test_layer1_input)
    test_error = T.mean(T.neq(test_y_pred, y))
    test_model_all = theano.function([x,y], test_error) #return the accuracy  
    test_model_tag = theano.function([x],test_y_pred)   #return the tag it predict  this is for GEC, no need for y
    test_model_prob = theano.function([x],test_y_prob)  #return the tag it predict  this is for GEC, no need for y
    #######IMPORTANT######

    #start training over mini-batches
    print '... training.....'
    epoch = 0
    best_val_perf = 0
    val_perf = 0
    test_perf = 0       
    cost_epoch = 0    
    while (epoch < n_epochs):        
        epoch = epoch + 1
        if shuffle_batch:
            probar = 0
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
                #process the processBar
                progress(30,int(100*probar/float(n_train_batches-1) ),1)
                probar +=1

        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch = train_model(minibatch_index)  
                set_zero(zero_vec)

                progress(30,int(100*minibatch_index/float(n_train_batches-1) ),1)

        train_losses = [test_model(i) for i in xrange(n_train_batches)]
        train_perf = 1 - np.mean(train_losses)
        val_losses = [val_model(i) for i in xrange(n_val_batches)]
        val_perf = 1- np.mean(val_losses)                        
        print('epoch %i, train perf %f %%, val perf %f ' % (epoch, train_perf * 100., val_perf*100.))
        if val_perf >= best_val_perf:
            best_val_perf = val_perf
            # Predict Work!
            test_loss = test_model_all(test_set_x,test_set_y)        
            test_tag  = test_model_tag(test_set_x)
            test_prob  = test_model_prob(test_set_x)
            test_perf = 1- test_loss         
            
    return test_perf,test_tag,test_prob

def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
        
def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates 

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)
    
def safe_update(dict_to, dict_from):
    """
    re-make update dictionary for safe updating
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to
    
def get_idx_from_sent(words, word_idx_map, max_l, filter_h):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
   
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

def make_idx_data(revs_train,revs_test, word_idx_map, max_l, filter_h,ET):
    """
    Transforms sentences into a 2-d matrix.
    """
    train_pre,train_data, test_pre,test_data = [], [], [], []

    train_pre,train_data = getTrainTest_idx_data(revs_train,word_idx_map, max_l, filter_h,ET)
    test_pre,test_data = getTrainTest_idx_data(revs_test,word_idx_map, max_l, filter_h,ET)
 
    train_data = np.array(train_data,dtype="int")
    test_data = np.array(test_data,dtype="int")

    return [train_data,test_data,train_pre,test_pre]     


def getTrainTest_idx_data(revs,word_idx_map, max_l, filter_h,ET):
    data = []
    prefix = []

    if ET=="-artordet":
        KV = AKV
    elif ET=="-prep":
        KV = PKV
    else:
        KV = NKV

    begin = 8
    
    for rev in revs:
        rev = rev.strip().split()
        words = rev[begin:]
        
        pre   = rev[:begin]  #nid,pid,sid,index,ti,sw,aw
        pre[-1] = KV.get(pre[-1],"0")
        pre[-2] = KV.get(pre[-2],"0")

        sent = get_idx_from_sent(words, word_idx_map, max_l, filter_h)
        sent.append(pre[-1])

        data.append(sent)
        prefix.append(pre)

    return prefix,data



if __name__=="__main__":
    """
    revs : the list the element is a dict named datum
    W    : the matrix of word-vector
    W2   : the matrix of word-vector by the vector 's value is random initial
    word_idx_map : the word's index in W  (for W2 there is no object record it)
    vocab: dict the key is word and the value is the time it occurs in the corpus

    """

    # Common paramether!!
    max_l = 8
    filter_h = 5

    if len(sys.argv)!=6:
        print "python command [-artordet|-prep|-nn] [-nonstatic|-static] [-rand|-word2vec] [vecotrL] [traintime]"
        assert False

    et = sys.argv[1]
    mode= sys.argv[2] # static or not
    word_vectors = sys.argv[3]   # random or not 
    vectorL = int(sys.argv[4])
    traintime = int(sys.argv[5])
    ET = et


    if et=="-artordet":
        input_file = "tmp/artordet.data"
        testRes = configure.fCNNResultArt
        classNum = 3
        print "The Vector lenght is %s and classNum is %s" %(vectorL,classNum)
    elif et=="-prep":
        input_file = "tmp/prep.data"
        testRes = configure.fCNNResultPrep
        classNum = 9
        print "The Vector lenght is %s and classNum is %s" %(vectorL,classNum)
    elif et=="-nn":
        input_file = "tmp/nn.data"
        testRes = configure.fCNNResultNn
        classNum = 2
        print "The Vector lenght is %s and classNum is %s" %(vectorL,classNum)
    else:
        print """Please input the error type: -artordet -prep
                 Please input the mode      : -nonstatic -static
                 please input the vector init: -word2vec -rand
              """
        assert(False)


    print "loading data...",

    x = cPickle.load(open(input_file,"rb"))
    revs_train,revs_test, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4], x[5]
    print "data loaded!"


    if mode=="-nonstatic":
        print "model architecture: CNN-non-static"
        non_static=True
    elif mode=="-static":
        print "model architecture: CNN-static"
        non_static=False
    execfile("conv_net_classes4GEC.py")   # note that we have change the tag Here! 
    if word_vectors=="-rand":
        print "using: random vectors"
        U = W2
    elif word_vectors=="-word2vec":
        print "using: word2vec vectors"
        U = W
    results = []

    datasets = make_idx_data(revs_train,revs_test, word_idx_map, max_l, filter_h,ET)

    perf,test_tag,test_prob = train_conv_net(datasets,
                          U,
                          img_w=vectorL, 
                          lr_decay=0.95,
                          filter_hs=[3,4,5,6],
                          conv_non_linear="relu",
                          hidden_units=[100,classNum],     # 100
                          shuffle_batch=True,               # is change the order of instance
                          n_epochs=traintime,                      # determine the time to loop 
                          sqr_norm_lim=9,
                          non_static=non_static,
                          batch_size=50,
                          dropout_rate=[0.5])



    test_prefix = datasets[3]
    res = list()
    # output the prefix information, tag and the probablity!!
    for item in zip(test_prefix,test_tag,test_prob):
        prob = [str(p) for p in item[2]]
        res.append("\t".join(item[0][:-1])+"\t"+str(item[1])+"\t"+"\t".join(prob)+"\n")

    # Save the file!
    open(testRes,"w").writelines(res)

    print "Finish it !"
