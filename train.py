import numpy as np

def import_data():
    X=np.genfromtxt("train_X_nb.csv",dtype=str,delimiter="\n")
    Y=np.genfromtxt("train_Y_nb.csv",dtype=str,delimiter=",")
    return X,Y

def remove_spl_chars_except_space(s):
    i = 0
    s_with_no_spl_chars = ""
    # using ASCII Values of characters
    for i in range(len(s)):
        if ord('A') <= ord(s[i]) <= ord('Z') or ord('a') <= ord(s[i]) <= ord('z') or ord(s[i]) == ord(' '):
            s_with_no_spl_chars += s[i]
    return s_with_no_spl_chars
def preprocessing(s):
    s = remove_spl_chars_except_space(s)
    s =' '.join(s.split())
    return s

def class_wise_words_frequency_dict(X, Y):
    class_wise_frequency_dict = dict()
    for i in range(len(X)):
        words = X[i].split()
        for token_word in words:
            y = Y[i]
            if y not in class_wise_frequency_dict:
                class_wise_frequency_dict[y] = dict()
            if token_word not in class_wise_frequency_dict[y]:
                class_wise_frequency_dict[y][token_word] = 0
            class_wise_frequency_dict[y][token_word] += 1
    return class_wise_frequency_dict

def compute_prior_probabilities(Y):
    Y=list(Y)
    classes = list(set(Y))
    n_docs = len(Y)
    prior_probabilities = dict()
    for c in classes:
        prior_probabilities[c] = Y.count(c) / n_docs
    return prior_probabilities


def get_class_wise_denominators_likelihood(X, Y):
    class_wise_frequency_dict = class_wise_words_frequency_dict(X, Y)
    class_wise_denominators = dict()
    vocabulary = []
    for c in classes:
        frequency_dict = class_wise_frequency_dict[c]
        class_wise_denominators[c] = sum(list(frequency_dict.values()))
        vocabulary += list(frequency_dict.keys())
    vocabulary = list(set(vocabulary))
    for c in classes:
        class_wise_denominators[c] += len(vocabulary)
    return class_wise_denominators

def compute_likelihood(test_X, c):
    likelihood = 0
    words = test_X.split()
    for word in words:
        count = 0
        words_frequency = class_wise_frequency_dict[c]
        if word in words_frequency:
            count = class_wise_frequency_dict[c][word]
        likelihood += np.log((count + 1)/class_wise_denominators[c])
    return likelihood

def predict(test_X):
    best_p = -99999
    best_c = -1
    for c in classes:
        p = compute_likelihood(test_X, c) + np.log(prior_probabilities[c])
        if p > best_p:
            best_p = p
            best_c = c
    return best_c

X,Y=import_data()

classes=list(set(Y))
classes.sort()
for k in range(len(X)):
    X[k]=preprocessing(X[k])
validate_X=X[:200]
validate_Y=Y[:200]
X=X[200:]
Y=Y[200:]
class_wise_frequency_dict=class_wise_words_frequency_dict(X,Y)
class_wise_denominators=get_class_wise_denominators_likelihood(X,Y)
prior_probabilities=compute_prior_probabilities(Y)
l=[]
for i in range(len(validate_X)):
    l.append(predict(validate_X[i]))
print(len(l))

