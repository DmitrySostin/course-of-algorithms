"""  Main file algoritm  """
import pygame
import json
from deyksty import find_shortest_path


fps = 30                                            # Количество кадров в секунду
H = 1080                                            # Ширина экрана
W = 700                                             # Высота экрана
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)




def main():
    """Start function"""
    with open("points.json", "r") as f:
        points = json.load(f)["points"]               # Графы
    with open("points.json", "r") as f:
        orders = json.load(f)["orders"]               # Параметры груза
    with open("points.json", "r") as f:
        vehicles = json.load(f)["vehicles"]           # Параметры транспортного средства


    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((H, W))
    pygame.display.set_caption("Planet Express")
    pygame.display.set_icon(pygame.image.load("statick\icon.png"))

    screen.fill(WHITE)
    font = pygame.font.SysFont('couriernew', 10)
   
    # прориссовка графов поинтов и рёбер пути соеденения
    for i in range(16):
        pygame.draw.circle(screen, GREEN, (points[i]["x"], points[i]["y"]), 15, width=3)
        text = font.render(points[i]['name'], True, BLUE)
        screen.blit(text, (points[i]["x"] + 25, points[i]["y"] -5))
        #print(points[i]['name'], points[i]["x"], points[i]["y"])
        for coint_id in points[i]['connections']:
            if coint_id is not None:
                print(points[i]['name'], '-', coint_id)
                for to_point in points:
                    if to_point['id'] == coint_id:
                        pygame.draw.line(screen, BLUE, (points[i]["x"], points[i]["y"]), (to_point['x'], to_point['y']), 1)

    

    

        
    

    run = True
    while run:
        """1-й Раздел ----- Обработка событий -----------"""
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                run = False
        
        pygame.display.flip()

    """2-й Раздел ----- Логика игры ------------------"""



    """# 3-й Раздел -----  Отображение графики ---------"""


    clock.tick(fps)
    pygame.quit()


if __name__ == "__main__":
    main()


        # Пример работы:
    start = "S"  # Склад (Earth)
    end = "A"    # Точка доставки (Sadr)

    # Вычисляем маршрут
    route = find_shortest_path(start, end)

    # Результат:
    print(f"Кратчайший путь от {start} до {end}: {route}")
