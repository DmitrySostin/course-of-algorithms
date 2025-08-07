"""  Main file algoritm  """
import pygame


fps = 30                                            #  количество кадров в секунду


def main():
    """Start function"""
    pygame.init()
    clock = pygame.time.Clock()
    pygame.display.set_mode((1080,700))
    pygame.display.set_caption("Planet Express")
    pygame.display.set_icon(pygame.image.load("statick\icon.png"))


    

    run = True
    while run:
        """1-й Раздел ----- Обработка событий -----------"""
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                run = False
            if e.type == pygame.KEYDOWN:
                my_key = e.key
                my_mod = e.mod
                print(f"Нажата клавиша: {pygame.key.name(my_key)}")
                print(f"Модификатор: {my_mod}")
            if e.type == pygame.MOUSEBUTTONDOWN:
                my_pos = e.pos
                my_button = e.button
                print(f"Позиция мыши: {my_pos}")
                print(f"Идентификатор кнопки мыши: {my_button}")


    """2-й Раздел ----- Логика игры ------------------"""



    """# 3-й Раздел -----  Отображение графики ---------"""
    clock.tick(fps)
    pygame.quit()


if __name__ == "__main__":
    main()
