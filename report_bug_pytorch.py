ex_label = y_tens[0]
# ex_label is tensor(3.)
target = torch.zeros(1, n_classes).scatter_(0, ex_label, 1.)