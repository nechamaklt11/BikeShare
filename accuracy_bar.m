function accuracy_bar(acc, avg_acc, bar_title)
figure
bar(acc,'m')
ylim([70 100]); xlabel('fold'); ylabel('accurecy (%)')
hold on
line([0.5 5.5],[mean(acc) mean(acc)],'Color','k','LineWidth',1)
line([0.5 5.5],[avg_acc, avg_acc],'Color','b','LineWidth',1)
legend('accuracy per fold','average accuray','average cross validation accuracy')
title(bar_title)
hold off
