def attention_new():
    # reshape the image representation
    img = Input(shape=(1,feat_dim, w, w))
    img_reshape = Reshape((feat_dim, w * w))(img)
    img_permute = Permute((2, 1))(img_reshape)

    # word-guided visual attention 
    img_permute_reshape = TimeDistributed(RepeatVector(sent_maxlen))(img_permute) 
    img_permute_reshape = Permute((2, 1, 3))(img_permute_reshape) 
    w_repeat = TimeDistributed(RepeatVector(w*w))(w_c_feature) 
    w_repeat = TimeDistributed(TimeDistributed(Dense(final_w_emb_dim)))(w_repeat)
    img_permute_reshape = TimeDistributed(TimeDistributed(Dense(final_w_emb_dim)))(img_permute_reshape)
    img_w_merge = merge([img_permute_reshape, w_repeat], mode='concat') 

    att_w = TimeDistributed(Activation('tanh'))(img_w_merge)
    att_w = TimeDistributed(TimeDistributed(Dense(1)))(att_w) 
    att_w = TimeDistributed(Flatten())(att_w) 
    att_w_probability = Activation('softmax')(att_w) 

    img_permute_r = TimeDistributed(Dense(final_w_emb_dim))(img_permute)
    img_new = merge([att_w_probability, img_permute_r], mode='dot', dot_axes=(2,1)) 


    # image-guided textual attention
    img_new_dense = TimeDistributed(Dense(final_w_emb_dim))(img_new)  
    img_new_rep = TimeDistributed(RepeatVector(sent_maxlen))(img_new_dense) 

    tweet_dense = TimeDistributed(Dense(final_w_emb_dim))(w_c_feature) 
    tweet_dense1 = Flatten()(tweet_dense)
    tweet_rep = RepeatVector(sent_maxlen)(tweet_dense1) 
    tweet_rep = Reshape((sent_maxlen, sent_maxlen, final_w_emb_dim))(tweet_rep)

    att_img = merge([img_new_rep, tweet_rep], mode='concat') 
    att_img = TimeDistributed(Activation('tanh')) (att_img) 
    att_img = TimeDistributed(TimeDistributed(Dense(1)))(att_img) 
    att_img = TimeDistributed(Flatten())(att_img) 
    att_img_probability = Activation('softmax')(att_img)

    tweet_new = merge([att_img_probability, tweet_dense], mode='dot', dot_axes=(2, 1)) 

    img_new_resize = TimeDistributed(Dense(final_w_emb_dim, activation='tanh'))(img_new) 
    tweet_new_resize = TimeDistributed(Dense(final_w_emb_dim, activation='tanh'))(tweet_new) 


    # gate -> img new
    merge_img_w = merge([img_new_resize, tweet_new_resize], mode='sum')
    gate_img = TimeDistributed(Dense(1, activation='sigmoid'))(merge_img_w)
    gate_img = TimeDistributed(RepeatVector(final_w_emb_dim))(gate_img)  
    gate_img = TimeDistributed(Flatten())(gate_img) 
    part_new_img = merge([gate_img, img_new_resize], mode='mul') 


    #gate -> tweet new
    gate_tweet = Lambda(lambda_rev_gate, output_shape=(sent_maxlen, final_w_emb_dim))(gate_img)
    part_new_tweet = merge([gate_tweet, tweet_new_resize], mode='mul')
    
    part_img_w = merge([part_new_img, part_new_tweet], mode='concat')
    part_img_w = TimeDistributed(Dense(final_w_emb_dim))(part_img_w)


    #gate -> multimodal feature
    gate_merg = TimeDistributed(Dense(1, activation='sigmoid'))(part_img_w)
    gate_merg = TimeDistributed(RepeatVector(final_w_emb_dim))(gate_merg)  
    gate_merg = TimeDistributed(Flatten())(gate_merg) 
    part_sample = merge([gate_merg, part_img_w], mode='mul')

    w_c_emb = TimeDistributed(Dense(final_w_emb_dim))(w_c_feature) 

    merge_multimodal_w = merge([part_sample, w_c_emb], mode='concat') 
    multimodal_w_feature = TimeDistributed(Dense(num_classes))(merge_multimodal_w)
	