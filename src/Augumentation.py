##modify the aug method on the one image
@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    ## Define augmentation functions (add)
    tta_transforms = [
        #torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        torchvision.transforms.RandomGrayscale(p=0.2),
        
    ]
    ## Define denormalization transform (add)
    denorm_transform = torchvision.transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
            )
    ## Create a composite transform that applies all the augmentation functions (add)
    composite_transform = torchvision.transforms.Compose(tta_transforms)

    # switch to evaluation mode
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        ## apply test-time augmentation with the composite transform (update)
        with torch.cuda.amp.autocast():
            output = model(composite_transform(images))
            
            
            
        ### all to one for 10, 20, 30, 40 times, an then get the mean.ls
        
        ## apply test-time augmentation (add)
        output_list = []
        original = model(images)
        output_list.append(original)
        for i in range(20):
            with torch.cuda.amp.autocast():
                #cur_output = model(composite_transform(images))
                ## Denormalize images (add)
                denorm = denorm_transform(images.clone())
                ## Apply TTA transforms (add)
                tta_images = composite_transform(denorm)
                ## Normalize again (add)
                tta_images = torchvision.transforms.functional.normalize(tta_images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                cur_output = model(tta_images)
                #print("cur_out: ", cur_output)
            output_list.append(cur_output)

        # take the average of model predictions from different augmentations
        output = torch.mean(torch.stack(output_list), dim=0)
        ###

        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
