'''
평가
'''
with torch.no_grad():
    predicted_data_list = []
    label_list = []
    for i, samples in enumerate(test_dataloader):
        x_test, y_test = samples
        x_test = x_test.view(-1, sequence_length, input_size).to(device)
        y_test = y_test.to(device)
        
        prediction = model(x_test)
        predicted_data_list.append(prediction.tolist())
        label_list.append(y_test.tolist())#.cpu().data.numpy())
        loss = criterion(prediction, y_test)
        
predicted_data_np = np.array(sum(sum(predicted_data_list, []), []))
label_np = np.array(sum(sum(label_list, []), []))