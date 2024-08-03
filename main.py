from maskexp.model.train import train_velocitymlm, continue_velocitymlm
import maskexp.model.train_ordinal as train_ordinal
import maskexp.model.train_raw as train_raw

if __name__ == '__main__':
    # train_velocitymlm()
    # continue_velocitymlm()
    train_ordinal.train_velocitymlm()
