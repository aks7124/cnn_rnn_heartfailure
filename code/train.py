
from CNN import ModelProcessor as CNNModel
from RETAIN import ModelProcessor as RETAINModel

def process():
    cnnModel =  CNNModel(n_epochs=10, batch_size=10)
    print('****Training CNN model.****')
    cnnModel.train()

    # extract the trained vectors, dimentions can be specified as below
    # dim=0 (default) gets all trained vectors
    (vectors, _) = cnnModel.get_vectors()

    retainModel = RETAINModel(n_epochs=10, batch_size=20, vectors=vectors)
    print('****Training RETAIN model.****')
    retainModel.train()

if __name__ == '__main__':
    process()    
    print('Done!')

