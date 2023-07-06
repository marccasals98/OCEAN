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
    """
    def __init__(self, labels: torch.Tensor, outputs: torch.Tensor, device) -> None:
        self.device = device
        self.outputs = outputs # we need this to know the number of classes.
        self.preds = outputs.argmax(-1)
        #self.preds = self.preds.to(self.device)
        self.labels = labels.argmax(-1, keepdim=True)
        self.labels = labels
    
    def precision(self) -> float:
        # Torch made a library that needs to be in CPU?
        #self.outputs = self.outputs.to('cpu')
        #self.preds= self.preds.to('cpu')
        self.labels= self.labels.to('cpu')
        print("labels:", self.labels)
        print("outputs:", self.preds)
        precision = tm.Precision(task='multiclass',
                                 num_classes=self.outputs.shape[1],) # the number of classes is properly calculated.
        self.precision = precision(self.labels, self.preds)
    
    def print_metrics(self):
        print(f"Precision: {self.precision}")

    



