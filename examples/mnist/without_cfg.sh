
python -m train_loop.train null   train.n_total_steps=100 train.save_dir='/tmp/dd'  model.name="examples.mnist.mnist.MNISTClassifier" dataset.name=examples.mnist.mnist.MNISTDataset losses.default_loss.function_name="cross_entropy_classification_loss"  losses.default_loss.tgt_key=gt_class_id  losses.default_loss.src_key=pred_logits  sanity=True 
