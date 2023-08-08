import torch
import torchmetrics as tm

def accuracy(labels: torch.Tensor, outputs: torch.Tensor) -> int:
    '''
    This function returns the number of coincidences that happen in two arrays of the same length.

    Arguments:
    ----------
    labels : torch.Tensor [batch, num_classes]
        The ground truth of the classes.
    outputs : torch.Tensor [batch, num_classes]
        The model prediction of the most likely class.
    
    Returns:
    --------
    acum : int
        The number of coincidences.
    '''
    # preds = outputs.argmax(-1, keepdim=True)
    preds = outputs.argmax(-1)
    labels = labels.argmax(-1, keepdim=True) # bc we have done one_hot encoding.
    # label shape [batch, 1], outputs shape [batch, 1]
    acum = preds.eq(labels.view_as(preds)).sum().item() # sums the times both arrays coincide.
    return acum

class Metrics():
    """
    This class is used to calculate the metrics of the model.
    
    Arguments:
    ----------
    labels : torch.Tensor [batch, num_classes]
        The ground truth of the classes.
    outputs : torch.Tensor [batch, num_classes]
        The model prediction of the most likely class.
    config : dict
        The dictionary with the run configuration. 
    device : torch.device
        The device where the model is running
    
    Methods:
    --------
    tensor_transformation: self -> None
        This function transforms the tensors to the correct shape for the metrics.
        batch_size x num_classes -> batch_size
    compute_precision: self -> None
        Computes the precision of the classifier
    compute_recall: self -> None
        Computes the recall of the classifier
    compute_f1: self -> None 
        Computes the F1Score
    compute_metrics: self -> None
        Computes all the metrics at once.
    """
    def __init__(self, labels: torch.Tensor, outputs: torch.Tensor, config: dict, device) -> None:
        self.labels = labels
        self.outputs = outputs
        self.device = device
        self.config = config
        Metrics.tensor_transformation(self)
    
    def tensor_transformation(self)-> None:
        '''
        This function transforms the tensors to the correct shape for the metrics.

        batch_size x num_classes -> batch_size

        '''
        self.labels = self.labels.argmax(-1, keepdim=True)
        self.outputs = self.outputs.argmax(-1, keepdim=True)
    
    def compute_precision(self) -> None:
        """
        Computes the precision.

        P = TP/(TP+FP)
        """
        precision = tm.Precision(task='multiclass',
                                 num_classes=len(self.config['species']),
                                 average='macro').to(self.device) # the dimension 1 is the number of classes.
        self.precision = precision(self.labels, self.outputs)
    
    def compute_recall(self) -> None:
        """
        Computes the recall.

        R = TP/(TP+FN)
        """
        recall = tm.Recall(task='multiclass',
                           num_classes=len(self.config['species']),
                           average='macro').to(self.device)
        self.recall = recall(self.labels, self.outputs)
    
    def compute_f1(self) -> None:
        """
        Computes the F1-Score.

        F1 = 2P*R/(P*R)
        """
        f1 = tm.F1Score(task='multiclass',
                   num_classes=len(self.config['species'])).to(self.device)
        self.f1 = f1(self.labels, self.outputs)
    
    def compute_metrics(self):
        """
        Compute all metrics at the same time.
        """
        self.compute_precision()
        self.compute_recall()
        self.compute_f1()

    



