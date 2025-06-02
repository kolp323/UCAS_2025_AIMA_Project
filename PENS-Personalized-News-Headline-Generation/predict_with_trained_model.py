from pensmodule.Generator import *
from pensmodule.Generator.eval import predict
from pensmodule.UserEncoder import NRMS


def load_model_from_ckpt(path):
    checkpoint = torch.load(path, weights_only= False)
    model = checkpoint['model']
    if torch.cuda.device_count() > 1:
        print('multiple gpu training')
        model = nn.DataParallel(model)
    return model

if __name__ == '__main__':
    embedding_matrix = np.load('./data2/embedding_matrix.npy')

    device = torch.device('cuda:0')

    usermodel = NRMS(embedding_matrix)
    usermodel.load_state_dict(torch.load('./runs/userencoder/NAML-2.pkl', ))
    usermodel = usermodel.to(device)
    usermodel.eval()

    model_path = './runs/seq2seq/exp/checkpoint_train_mod4_step_2000.pth'

    news_scoring = np.load('./data2/news_scoring2.npy')
    sources = np.load('./data2/sources.npy')
    with open('./data2/TestUsers.pkl', 'rb') as f:
        TestUsers = pickle.load(f)
    with open('./data2/TestSamples.pkl', 'rb') as f:
        TestSamples = pickle.load(f)
    i_dset = TestImpressionDataset(news_scoring, sources, TestUsers, TestSamples)
    test_iter = DataLoader(i_dset, batch_size=16, shuffle=False)

    with open('./data2/dict.pkl', 'rb') as f:
        news_index,category_dict,word_dict = pickle.load(f)
    index2word = {}
    
    for k,v in word_dict.items():
        index2word[v] = k
        model = load_model_from_ckpt(model_path).to(device)
        model.eval()

    refs, hyps, scores1, scores2, scoresf = predict(
        usermodel, model, test_iter, device, index2word, beam=False, beam_size=3, eos_id=2
    )

    with open('./prediction/predictions.txt', 'w', encoding='utf-8') as f:
        for ref, hyp in zip(refs, hyps):
            f.write(f"REF: {ref}\nHYP: {hyp}\n\n")