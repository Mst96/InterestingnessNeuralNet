shuffledArray = output(randperm(size(output,1)),:);
data_train = shuffledArray(1:3500,1:31);
data_test = shuffledArray(3501:7000,1:31);

num_correct = 0;

x = [6.2 3.1 4.5 1.2];
distances = zeros(3500,1);
predictions = zeros(3500,1);

for i_test = 1:3500
% compute distances for each test sample distances = zeros(75,1);
    for i_train=1:3500
        diff = data_train(i_train,1:30) - data_test(i_test,1:30);
        distances(i_train) = sqrt(sum(diff.*diff));
    end
    distances = [distances (1:3500)']; 
    distances = sortrows(distances,1);
    
    class_predict = mode([data_train(distances(1,2),31)... 
                          data_train(distances(2,2),31)...
                          data_train(distances(3,2),31)...
                          data_train(distances(4,2),31)...
                          data_train(distances(5,2),31)]);
    predictions(i_test) = class_predict;
    if (class_predict == data_test(i_test,31))
        num_correct = num_correct+1;
    end
end

% accuracy rate
acc = num_correct/75
% print(num_correct/75)