function accuracy = evaluate_net(targets,predictions)
predictions(predictions<0.5)=0; predictions(predictions>=0.5)=1;
diff = targets-predictions;
num_hits= sum(diff==0);
accuracy = (num_hits/length(targets))*100;


