        
        # changes in dirchlet kl loss
        print("Given dim alphas: ", alphas.shape, " labels: ", labels.shape if labels is not None else '')
        k = alphas.shape[1] # num_classes
        if labels is None:
            # ood sample, set all alphas to 1 to get a flat simplex
            target_alphas = torch.ones_like(alphas)
        else:
            # in domain sample
            precision = 1000 # alpha_0 in paper
            target_alphas = torch.ones_like(alphas) * self.concentration # consider concentration as epsilon smoothing param in paper
            target_alphas = torch.clone(target_alphas).scatter_(1, labels[:, None],
                                                               1-(k-1)*self.concentration)
            target_alphas *= precision
        print("target alphas:", target_alphas[0])