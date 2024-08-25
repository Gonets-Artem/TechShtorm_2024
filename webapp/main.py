import streamlit as st
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class MyHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if (event.src_path == "./activity.csv"):
            update_line(team_filter, placeholder)

        if (event.src_path == "./event_stats.csv"):
            update_diagram_table(team_filter, placeholder)

        if (event.src_path == "./goal_stats.csv"):
            update_metrics(team_filter, placeholder)

        if (event.src_path == "./keep_stats"):
            update_diagram_table(team_filter, placeholder)



def update_metrics(team_filter, placeholder):
    goals = pd.read_csv('goal_stats.csv', index_col=["teams"])
    with placeholder.container():
        goals_filter = goals[goals.index == team_filter]
        kpi1, kpi2 = st.columns(2)
        kpi1.metric(
            label="Голы 🥅",
            value=round(goals_filter['goals'])
        )

        kpi2.metric(
            label="Точность бросков 💯🔥",
            value=round(goals_filter['goals'] / goals_filter['attempts'] * 100, 2),
        )


def update_diagram_table(team_filter, placeholder):
    keeps = pd.read_csv('keep_stats.csv')
    # Круговая диаграмма
    fig_pie = px.pie(keeps, values='duration', names='teams',
                     width=400, height=400,
                     labels={"teams": "Команды", "duration": "Продолжительность (сек)"},
                     category_orders={"teams": ["Первая команда", "Вторая команда"]}
                     )

    events = pd.read_csv('event_stats.csv')
    with placeholder.container():
        if team_filter == 'all':
            table_filter = events
        else:
            table_filter = events[events['team'] == team_filter]
        table = go.Figure(data=[go.Table(header=dict(values=['<b>Событие</b>',
                                                             '<b>Команда</b>',
                                                             '<b>Время</b>']),
                                         cells=dict(values=[table_filter['event'],
                                                            table_filter['team'],
                                                            table_filter['timing']],
                                                    align='left'))])

    col1, col2 = st.columns(2, gap='medium')
    with col1:
        st.write('Процент владения шайбой')
        st.plotly_chart(fig_pie)
    with col2:
        st.write('Зафиксированные события')
        st.plotly_chart(table, use_container_width=True)


def update_line(team_filter, placeholder):
    activity = agregate(pd.read_csv('activity.csv'))
    with placeholder.container():
        if team_filter == 'all':
            activity_filter = activity
        else:
            activity_filter = activity[activity['team'] == team_filter + '_q']
        fig = px.line(activity_filter, x='time', y='score', color='team')
        fig.update_layout(title_text='Активность игроков на протяжении матча')
    st.plotly_chart(fig, use_container_width=True)


def agregate(data):
    res = []
    for team in ['team1_q', 'team2_q']:
        filtered = data[['time', 'event', 'lead', team]]

        lead_flag = int(team[-3])
        for d in filtered.values:
            score = 0
            if d[1] == 'goal':
                score += 5 * (d[2] == lead_flag) + 3 + d[3] * 0.25
            elif d[1] == 'try':
                score += 4.7 + (d[2] == lead_flag) + ((d[2] == lead_flag) - 1) * 1.2 + d[3] * 0.25
            elif d[1] == 'mess':
                score += 4 + (d[2] == lead_flag) * 0.8 + 1 + d[3] * 0.25
            elif d[1] == 'keep':
                score += 3.4 + (d[2] == lead_flag) * 0.5 + d[3] * 0.25
            elif d[1] == 'other':
                score += 2.8 + (d[2] == lead_flag) + d[3] * 0.25
            elif d[1] == 'start':
                score += 3 + (d[2] == lead_flag)
            res.append([d[0], team, score * score])

    return pd.DataFrame(res, columns=['time', 'team', 'score']).sort_values(by='time').reset_index(drop=True)


st.set_page_config(page_title="Распознавание событий хоккейного матча",
                           page_icon="🏒", )
st.title("Распознавание игровых событий хоккейного матча на базе ИИ")

event_handler = MyHandler()
observer = Observer()
observer.schedule(event_handler, path='.', recursive=False)
observer.start()


uploaded_file = st.file_uploader("Загрузите видео хоккейного матча в формате MP4:", type=["mp4"],
                                         accept_multiple_files=False)
link = st.text_input("Или вставьте ссылку:", placeholder="https://example.com/not-youtube.mp4")

if uploaded_file is not None or link:
    if link:
        with st.spinner('Пожалуйста, подождите...'):
            time.sleep(3)
        st.video(link, autoplay=True, muted=True)
        st.write("Файл успешно загружен!")
    elif uploaded_file:
        bytes_data = uploaded_file.read()
        st.video(bytes_data, autoplay=True, muted=True)
        st.write("Файл успешно загружен!")
    else:
        st.write("Что-то пошло не так")

    st.markdown("### Статистика матча:")
    with st.spinner('Пожалуйста, подождите...'):
        time.sleep(1)

    goals = pd.read_csv('goal_stats.csv', index_col=["teams"])
    team_filter = st.selectbox("Выберите команду", goals.index)
    placeholder = st.empty()

    update_metrics(team_filter, placeholder)
    update_diagram_table(team_filter, placeholder)
    update_line(team_filter, placeholder)

#observer.stop()
