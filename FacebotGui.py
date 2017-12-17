'''
Takes in a question, and a trained model and returns the
model's answer for that question. 
'''

def get_output(question, sess, w2idx, model, metadata):

    import data_utils
    from datasets.facebook2 import data

    # get output for input phrase
    idx_q, idx_a = data.process_input(question, w2idx)
    gen = data_utils.rand_batch_gen(idx_q, idx_a, 1)
    input_ = gen.__next__()[0]
    output = model.predict(sess, input_)

    # return ouput phrase
    for ii, oi in zip(input_.T, output):
        q = data_utils.decode(sequence=ii, lookup=metadata['idx2w'], separator=' ')
        decoded = data_utils.decode(sequence=oi, lookup=metadata['idx2w'], separator=' ').split(' ')
        return ' '.join(decoded)
