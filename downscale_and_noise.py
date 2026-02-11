from PIL import Image
import numpy as np
import random
from perlin_noise import PerlinNoise
from sklearn.preprocessing import minmax_scale
import click


# todo recode from https://editor.p5js.org/Pi_p5/sketches/qAqoieAhx

def add_noise_to_img(img, dist_from_center=50, amount_of_noise=50):
    '''
    다운스케일된 작은 이미지의 가장자리 영역을 확률적으로 제거하고,
    제거된 위치를 mask로 반환하는 함수.

    Parameters
    ----------
    img : PIL.Image
        다운스케일된 작은 이미지
    dist_from_center : int
        중앙 기준으로 노이즈가 적용될 영역 비율 (0~100)
    amount_of_noise : int
        노이즈 제거 강도 (확률 시작값)

    Returns
    -------
    img_arr : np.ndarray
        가장자리 일부가 제거된 이미지 배열
    mask : np.ndarray
        제거된 위치를 표시한 마스크 (255 = 제거된 위치)  
  '''
  img_arr = np.array(img)

  # 제거된 위치를 기록할 마스크 배열 생성
  # 초기값은 0 (제거 안 됨)  
  mask = np.zeros_like(img_arr)

  # 정사각형 이미지 한 변 길이
  dim = len(img_arr)

  prob_noise = 1
  dist_noise = 0.5
  x,y = dim,dim

  # 연산 안정성을 위해 4의 배수로 정렬
  rx,ry = dim,dim
  rx -= rx % 4 
  ry -= ry % 4

  # 제거 영역 비율 계산
  # 예: 50/200 = 0.25 (이미지의 25% 영역)
  tr_size = dist_from_center/200.0  # [0,100]
  
  # 제거 확률 시작값
  tr_strt = amount_of_noise  #[0,90]

  # 왼쪽 영역 제거
  for i in range(rx):
    for j in range(int(ry*tr_size)):
      if random.randint(0,100) > tr_strt+(j/(ry*tr_size))*(100-tr_strt): # 중앙으로 갈수록 제거 확률 감소
        mask[i][j] = [255,255,255] # 제거 위치 표시
        img_arr[i][j] = [0,0,0]    # 픽셀 제거

  # 위쪽 영역 제거
  for i in range(int(rx*tr_size)):
    for j in range(ry):
      if random.randint(0,100) > tr_strt+(i/(rx*tr_size))*(100-tr_strt):
        mask[i][j] = [255,255,255]
        img_arr[i][j] = [0,0,0] 

  # 오른쪽 영역 제거
  for i in range(rx-1, -1, -1):
    for j in range(ry-1,int(ry-(ry*tr_size)), -1):
      if random.randint(0,100) > tr_strt+((ry-j)/(ry*tr_size))*(100-tr_strt):
        img_arr[i][j] = [0,0,0] 
        mask[i][j] = [255,255,255]

  # 아래쪽 영역 제거
  for i in range(rx-1,int(rx-(rx*tr_size)), -1):
    for j in range(ry-1,-1, -1):
      if random.randint(0,100) > tr_strt+((rx-i)/(rx*tr_size))*(100-tr_strt):
        img_arr[i][j] = [0,0,0] 
        mask[i][j] = [255,255,255]



  return img_arr, mask


downscale_factor = 4

noise11 = PerlinNoise(octaves=10)
noise12 = PerlinNoise(octaves=5)

noise21 = PerlinNoise(octaves=10)
noise22 = PerlinNoise(octaves=5)

noise31 = PerlinNoise(octaves=10)
noise32 = PerlinNoise(octaves=5)


def noise_mult_1(i,j, xpix=512,ypix=512):
  return noise11([i/xpix, j/ypix]) + 0.5 * noise12([i/xpix, j/ypix]) #+ 0.25 * noise3([i/xpix, j/ypix]) + 1.125 * noise4([i/xpix, j/ypix])

def noise_mult_2(i,j, xpix=512,ypix=512):
  return noise21([i/xpix, j/ypix]) + 0.5 * noise22([i/xpix, j/ypix]) #+ 0.25 * noise3([i/xpix, j/ypix]) + 1.125 * noise4([i/xpix, j/ypix])

def noise_mult_3(i,j, xpix=512,ypix=512):
  return noise31([i/xpix, j/ypix]) + 0.5 * noise32([i/xpix, j/ypix]) #+ 0.25 * noise3([i/xpix, j/ypix]) + 1.125 * noise4([i/xpix, j/ypix])

def get_mask_image(img, downscale_factor=4, noise_distance=20, noise_prob=0):
  """
    원본 이미지에서 중앙 삽입용 작은 이미지를 생성하고,
    해당 mask를 전체 크기 mask로 확장하는 함수.

    Returns
    -------
    full_mask : np.ndarray
        전체 크기 마스크 (512x512)
    noised_img : np.ndarray
        중앙에 삽입될 작은 이미지  
  """
  
  # 중앙에 삽입할 작은 이미지 생성
  img_downscaled = img.resize((int(img.size[0] / downscale_factor), int(img.size[1] / downscale_factor)))
    
  # 작은 이미지 가장자리 제거
  noised_img, mask = add_noise_to_img(img_downscaled, 20, 0)
  img_arr = np.array(img)

  # 중앙 영역 크기 계산
  small_size = int(len(img_arr) / downscale_factor)
    
  # 중앙 시작 좌표 계산 (정중앙 정렬)
  xy1_mask_img = int(len(img_arr) - small_size - ( ( len(img_arr) - (len(img_arr)/ downscale_factor)  ) / 2))
  xy2_mask_img = xy1_mask_img + small_size

  # 전체를 흰색(255)으로 초기화
  full_mask = np.zeros_like(np.array(img))
  full_mask.fill(255)

  # 중앙 영역에 작은 mask 삽입
  for i in range(len(full_mask)):
    for j in range(len(full_mask)):
      if i >= xy1_mask_img and j > xy1_mask_img and i < xy2_mask_img and j < xy2_mask_img:
        full_mask[i][j] = mask[i-xy1_mask_img][j-xy1_mask_img]

  return full_mask, noised_img



def get_init_image(noised_img, full_mask, xpix=512,ypix=512):
  """
    전체 Perlin Noise 이미지를 생성한 뒤,
    중앙 mask 영역만 noised_img로 교체하여
    diffusion 초기 입력 이미지를 생성하는 함수.    
  """
  click.echo('Generating noise ...')

  # 전체 Perlin Noise 생성
  pic = [[[noise_mult_1(i,j), noise_mult_2(i,j), noise_mult_3(i,j) ] for j in range(xpix)] for i in range(ypix)]
  
  click.echo('Noise generated !')

  # 0~255 범위로 정규화
  scaled_noise = minmax_scale(np.array(pic).flatten(), (0,255)).reshape((512,512, 3))
  scaled_noise = scaled_noise.astype(np.uint8)

  init_image = scaled_noise.copy()
  noised_img_arr = np.array(noised_img)

  # 중앙 좌표 재계산
  small_size = int(len(init_image) / downscale_factor)
  xy1_mask_img = int(len(init_image) - small_size - ( ( len(init_image) - (len(init_image)/ downscale_factor)  ) / 2))
  xy2_mask_img = xy1_mask_img + small_size

  # mask가 흰색이 아닌 영역만 교체
  for i in range(len(scaled_noise)):
    for j in range(len(scaled_noise)):
      if i >= xy1_mask_img and j > xy1_mask_img and i < xy2_mask_img and j < xy2_mask_img and list(full_mask[i][j]) != [255,255,255]:
        init_image[i][j] = noised_img_arr[i-xy1_mask_img][j-xy1_mask_img]
          
  return init_image


def get_init_mask_image(img, downscale_factor=4, noise_distance=20, noise_prob=0):
  full_mask, noised_image = get_mask_image(img, downscale_factor,noise_distance, noise_prob)
  init_image = get_init_image(noised_image, full_mask)
  return init_image, full_mask

@click.command()
@click.option('--input_image', help='Image to use as input. (512x512)')
@click.option('--output_init', default="./init.png", help='Path to save init image.')
@click.option('--output_mask', default="./mask.png", help='Path to save mask image.')
@click.option('--downscale_factor', default=4)
@click.option('--noise_distance', default=20)
@click.option('--noise_prob', default=0)

def cmd_get_init_mask_image(input_image,output_init, output_mask, 
                            downscale_factor,noise_distance, noise_prob):
  with open(input_image, 'rb') as f:
    base_image = Image.open(f)
    base_image.load()
  
  init_image, mask = get_init_mask_image(base_image,downscale_factor, noise_distance, noise_prob)
  click.echo('Saving images ...')

  Image.fromarray(init_image).save(output_init)
  Image.fromarray(mask).save(output_mask) 

  click.echo('Done ! :)')



if __name__ == '__main__':
  cmd_get_init_mask_image()






