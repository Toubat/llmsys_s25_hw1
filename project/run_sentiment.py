import random
import pdb

import embeddings

import sys
sys.path.append('../')
import minitorch
from minitorch.tensor import Tensor

from datasets.load import load_dataset

backend_name = "CudaKernelOps"

if backend_name == "CudaKernelOps":
    from minitorch.cuda_kernel_ops import CudaKernelOps
    BACKEND = minitorch.TensorBackend(CudaKernelOps)

BATCH = 10


def RParam(*shape):
    r = 0.1 * (minitorch.rand(shape, backend=BACKEND) - 0.5)
    return minitorch.Parameter(r)


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        
        # BEGIN ASSIGN1_3
        # 1. Initialize self.weights to be a random parameter of (in_size, out_size).
        # 2. Initialize self.bias to be a random parameter of (out_size)
        # 3. Set self.out_size to be out_size
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size
        # END ASSIGN1_3

    def forward(self, x: Tensor):
        
        batch, in_size = x.shape
        
        # BEGIN ASSIGN1_3
        # 1. Reshape the input x to be of size (batch, in_size)
        # 2. Reshape self.weights to be of size (in_size, self.out_size)
        # 3. Apply Matrix Multiplication on input x and self.weights, and reshape the output to be of size (batch, self.out_size)
        # 4. Add self.bias
        # HINT: You can use the view function of minitorch.tensor for reshape
        x = x.view(batch, in_size)
        w = self.weights.value.view(in_size, self.out_size)
        b = self.bias.value.view(1, self.out_size)

        return x @ w + b
        # END ASSIGN1_3

        
class Network(minitorch.Module):
    """
    Implement a MLP for SST-2 sentence sentiment classification.

    This model should implement the following procedure:

    1. Average over the sentence length.
    2. Apply a Linear layer to hidden_dim followed by a ReLU and Dropout.
    3. Apply a Linear to size C (number of classes).
    4. Apply a sigmoid.
    """

    def __init__(
        self,
        embedding_dim=50,
        hidden_dim=128,
        dropout_prob=0.5,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout_prob
                
        # BEGIN ASSIGN1_3
        # 1. Construct two linear layers: the first one is embedding_dim * hidden_dim, the second one is hidden_dim * 1
        self.linear1 = Linear(embedding_dim, hidden_dim)
        self.linear2 = Linear(hidden_dim, hidden_dim)
        self.out = Linear(hidden_dim, 1)
        # END ASSIGN1_3
        

    def forward(self, embeddings: Tensor):
        """
        embeddings tensor: [batch x sentence length x embedding dim]
        """
    
        # BEGIN ASSIGN1_3
        # 1. Average the embeddings on the sentence length dimension to obtain a tensor of (batch, embedding_dim)
        # 2. Apply the first linear layer
        # 3. Apply ReLU and dropout (with dropout probability=self.dropout_prob)
        # 4. Apply the second linear layer
        # 5. Apply sigmoid and reshape to (batch)
        # HINT: You can use minitorch.dropout for dropout, and minitorch.tensor.relu for ReLU
        
        batch, _, embed_dim = embeddings.shape
        assert embed_dim == self.embedding_dim
        
        x = embeddings.mean(1).view(batch, self.embedding_dim) # (batch, embedding_dim)
        x = self.linear1(x) # (batch, hidden_dim)
        x = minitorch.dropout(x.relu(), self.dropout_prob) # (batch, hidden_dim)
        x = self.linear2(x) # (batch, hidden_dim)
        x = minitorch.dropout(x.relu(), self.dropout_prob) # (batch, hidden_dim)
        x = self.out(x) # (batch, 1)
        x = x.sigmoid().view(batch) # (batch,)

        return x
        # END ASSIGN1_3


class BinaryCrossEntropyLoss(minitorch.Module):

    def forward(self, x: Tensor, y: Tensor):
        """
        x: output of the model of size (batch, 1)
        target: true labels of size (batch,)
        """
        
        batch = x.shape[0]
        assert x.shape == y.shape, f"Shape mismatch: x: {x.shape} vs y: {y.shape}"

        ones = minitorch.tensor([1.0] * batch, backend=BACKEND)
        loss = - y * x.log() + (y - ones) * (ones - x).log() # (batch,)
        loss = loss.mean()
        
        return loss


# Evaluation helper methods
def get_predictions_array(y_true, model_output):
    predictions_array = []
    model_output = model_output.view(model_output.shape[0])
    
    for j in range(model_output.shape[0]):
        true_label = y_true[j]
        logit = model_output[j]
        if logit > 0.5:
            predicted_label = 1.0
        else:
            predicted_label = 0
        predictions_array.append((true_label, predicted_label, logit))
    return predictions_array


def get_accuracy(predictions_array):
    correct = 0
    for (y_true, y_pred, logit) in predictions_array:
        if y_true == y_pred:
            correct += 1
    return correct / len(predictions_array)


best_val = 0.0


def default_log_fn(
    epoch,
    train_loss,
    train_accuracy,
    validation_predictions,
    validation_accuracy,
):
    global best_val
    best_val = (
        best_val if best_val > validation_accuracy[-1] else validation_accuracy[-1]
    )
    print(f"Epoch {epoch}, loss {train_loss}, train accuracy: {train_accuracy[-1]:.2%}")
    if len(validation_predictions) > 0:
        print(f"Validation accuracy: {validation_accuracy[-1]:.2%}")
        print(f"Best Valid accuracy: {best_val:.2%}")


class SentenceSentimentTrain:
    '''
        The trainer class of sentence sentiment classification
    '''
    def __init__(self):
        self.model = Network()

    def train(
        self,
        data_train,
        learning_rate,
        batch_size=BATCH,
        max_epochs=500,
        data_val=None,
        log_fn=default_log_fn,
    ):
        model = self.model

        (X_train, y_train) = data_train
        n_training_samples = len(X_train)
        
        losses = []
        train_accuracy = []
        validation_accuracy = []

        criterion = BinaryCrossEntropyLoss()
        optim_adam = minitorch.Adam(self.model.parameters(), learning_rate)
        optim_sgd = minitorch.SGD(self.model.parameters(), 0.1)

        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0
            n_batches = 0
            
            model.train()
            train_predictions = []
            batch_size = min(batch_size, n_training_samples)
            
            print(f"Epoch {epoch}")
            for batch_num, example_num in enumerate(
                range(0, n_training_samples, batch_size)
            ):  
                # BEGIN ASSIGN1_4
                # 1. Create x and y using minitorch.tensor function through our CudaKernelOps backend
                # 2. Set requires_grad=True for x and y
                # 3. Get the model output (as out)
                # 4. Calculate the loss using Binary Crossentropy Loss
                # 5. Call backward function of the loss
                # 6. Use Optimizer to take a gradient step
                X_batch = X_train[example_num : example_num + batch_size]
                y_batch = y_train[example_num : example_num + batch_size]

                x = minitorch.tensor(X_batch, requires_grad=True, backend=BACKEND)
                y = minitorch.tensor(y_batch, requires_grad=True, backend=BACKEND)
                
                out: Tensor = model(x) # (batch,)
                loss = criterion(out, y)

                optim = optim_adam if epoch < 20 else optim_sgd

                optim.zero_grad()
                loss.backward()
                optim.step()
                # END ASSIGN1_4
                
                # Save training results
                train_predictions += get_predictions_array(y, out)
                total_loss += loss[0]
                n_batches += 1
        
            # Evaluate on validation set at the end of the epoch
            validation_predictions = []
            if data_val is not None:
                (X_val, y_val) = data_val
                model.eval()
                
                # BEGIN ASSIGN1_4
                # 1. Create x and y using minitorch.tensor function through our CudaKernelOps backend
                # 2. Get the output of the model
                # 3. Obtain validation predictions using the get_predictions_array function, and add to the validation_predictions list
                # 4. Obtain the validation accuracy using the get_accuracy function, and add to the validation_accuracy list
                x = minitorch.tensor(X_val, requires_grad=True, backend=BACKEND)
                y = minitorch.tensor(y_val, requires_grad=True, backend=BACKEND)

                out: Tensor = model(x) # (batch,)
                validation_predictions = get_predictions_array(y, out)
                validation_accuracy.append(get_accuracy(validation_predictions))
                # END ASSIGN1_4
                
                model.train()

            train_accuracy.append(get_accuracy(train_predictions))
            losses.append(total_loss/n_batches)
            log_fn(
                epoch,
                total_loss/n_batches,
                train_accuracy,
                validation_predictions,
                validation_accuracy,
            )
        

def encode_sentences(
    dataset, N, max_sentence_len, embeddings_lookup, unk_embedding, unks
):
    Xs = []
    ys = []
    for sentence in dataset["sentence"][:N]:
        # pad with 0s to max sentence length in order to enable batching
        sentence_embedding = [[0] * embeddings_lookup.d_emb] * max_sentence_len
        for i, w in enumerate(sentence.split()):
            sentence_embedding[i] = [0] * embeddings_lookup.d_emb
            if w in embeddings_lookup:
                sentence_embedding[i][:] = embeddings_lookup.emb(w)
            else:
                # use random embedding for unks
                unks.add(w)
                sentence_embedding[i][:] = unk_embedding
        Xs.append(sentence_embedding)

    # load labels
    ys = dataset["label"][:N]
    return Xs, ys


def encode_sentiment_data(dataset, pretrained_embeddings, N_train, N_val=0):
    #  Determine max sentence length for padding
    max_sentence_len = 0
    for sentence in dataset["train"]["sentence"] + dataset["validation"]["sentence"]:
        max_sentence_len = max(max_sentence_len, len(sentence.split()))

    unks = set()
    unk_embedding = [
        0.1 * (random.random() - 0.5) for i in range(pretrained_embeddings.d_emb)
    ]

    X_train, y_train = encode_sentences(
        dataset["train"],
        N_train,
        max_sentence_len,
        pretrained_embeddings,
        unk_embedding,
        unks,
    )
    X_val, y_val = encode_sentences(
        dataset["validation"],
        N_val,
        max_sentence_len,
        pretrained_embeddings,
        unk_embedding,
        unks,
    )
    print(f"missing pre-trained embedding for {len(unks)} unknown words")

    return (X_train, y_train), (X_val, y_val)


if __name__ == "__main__":
    train_size = 450
    validation_size = 100
    learning_rate = 0.25
    max_epochs = 50
    embedding_dim = 50

    (X_train, y_train), (X_val, y_val) = encode_sentiment_data(
        load_dataset("glue", "sst2"),
        embeddings.GloveEmbedding("wikipedia_gigaword", d_emb=embedding_dim, show_progress=True),
        train_size,
        validation_size,
    )
    model_trainer = SentenceSentimentTrain()
    model_trainer.train(
        (X_train, y_train),
        learning_rate,
        max_epochs=max_epochs,
        data_val=(X_val, y_val),
    )
