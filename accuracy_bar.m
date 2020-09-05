function accuracy_bar(acc, avg_acc, bar_title)
figure
bar(acc,'m')
xlabel('fold'); ylabel('accurecy (%)')
ylim([70 90]);
hold on
line([0.5 5.5],[mean(acc) mean(acc)],'Color','k','LineWidth',1)
line([0.5 5.5],[avg_acc, avg_acc],'Color','b','LineWidth',1)
legend('accuracy per fold','average accuray','average cross validation accuracy')
title(bar_title)
hold off
