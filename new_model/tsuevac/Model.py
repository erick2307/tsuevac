from tsuevac import Environment
from tsuevac import Agent
from tsuevac import Evacuee
from tsuevac import Node
from tsuevac import Shelter
from tsuevac import setup
import random
import numpy as np
from pathlib import Path
import geopandas as gpd

class Model(object):
    """A Model class"""
    def __init__(self, case, nw_type, tsunami, verb=False):
        """Initialize the model"""
        self.clean()
        if not setup.REPLICATION:
            random.seed(setup.SEED)
            np.random.seed(setup.SEED)
            self.rng = np.random.default_rng(setup.SEED)
        self.setup()
        self.id = case
        self.tsunami = tsunami
        self.Env = Environment(self.bbox, nw_type, verb)
        self.Env.e_get_network_projection(self.Env.e_G, verb)
        self.nodes = self.Env.e_get_nodes(verb)
        self.edges = self.Env.e_edges
        self.create_node_agents()
        self.Env.e_load_aos(aos_file=setup.AOS_FILE, explore=False, verb=verb)
        self.step = 0

    def setup(self):
        p = Path(f'./cases/{setup.CASE}')
        try:
            p.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print('The directory already exists.')

        if setup.AOS_FILE is not None:
            gdf = gpd.read_file(setup.AOS_FILE)
            bbox = gdf.total_bounds
            self.bbox = {'north': bbox[3], 'south': bbox[1], 'east': bbox[2], 'west': bbox[0]}

    def create_node_agents(self):
        pass

    def initial_condition(self):
        pass

    def read_shelters_from_file(self):
        pass
        # df = pd.read_csv(filename)
        # gdf = gpd.GeoDataFrame(df.drop(['lat','lon'],axis=1),crs="EPSG:4326",
        # geometry=[Point(xy) for xy in zip(df.lon,df.lat)])
        # return gdf

    def clean(self):
        # try:
        #     Ambulances.count = 0
        #     Ambulances.ambs = []
        #     Hospitals.count = 0
        #     Hospitals.hosps = []
        #     Injuries.count = 0
        #     Injuries.injuries = []
        # except:
        pass

    # def assign_injury(self,verb=False):
    #     self.ambulance_standby_injury[0].a_g_obj = self.injury_wait_list[0] #傷病者の受け渡し（仮）
    #     self.ambulance_standby_injury[0].a_destination =  self.injury_wait_list[0].i_home#目的地の設定
    #     self.ambulance_standby_injury[0].a_mypath =
    # self.ambulance_standby_injury[0].find_path(self,self.Env.e_G,self.ambulance_standby_injury[0].a_current,self.ambulance_standby_injury[0].a_destination,verb=verb)#経路探索
    #     self.ambulance_standby_injury[0].a_state = 1 #moving
    #     self.ambulance_standby_injury[0].a_task = 1 #patient
    #     if verb:
    #         a = self.ambulance_standby_injury[0]
    #         i = self.injury_wait_list[0]
    #         print(f'{self.step}> A{a.a_uid} assigned to P{i.i_uid}')
    #     self.ambulance_standby_injury.pop(0)#受け渡し後のリストから削除
    #     self.injury_in_transport.append(self.injury_wait_list[0])
    #     self.injury_wait_list.pop(0)#受け渡し後のリストから削除
    #     if verb:
    #         print(f'{self.step}> Resources:{len(self.ambulance_standby_injury)}')
    #         print(f'{self.step}> Demands:{len(self.injury_wait_list)}')
    #         print('========')

    # def assign_hospital(self,max_iter,verb=False):
    #     h = rng.integers(low=0, high=len(self.hospitals), size=1)[0] #random assignment
    #     hosp = self.hospitals[h] #random hospital
    #     self.ambulance_standby_hospital[0].a_g_obj = hosp
    #     self.ambulance_standby_hospital[0].a_destination =  hosp.h_home#目的地の設定
    #     self.ambulance_standby_hospital[0].a_mypath =
    # self.ambulance_standby_hospital[0].find_path(self,self.Env.e_G,self.ambulance_standby_hospital[0].a_current,self.ambulance_standby_hospital[0].a_destination,verb=verb)#経路探索
    #     key = self.check_path(max_iter,self.ambulance_standby_hospital[0])
    #     if key:
    #         return key
    #     self.ambulance_standby_hospital[0].a_state = 1 #moving
    #     self.ambulance_standby_hospital[0].a_task = 2 #moving
    #     if verb:
    #         a = self.ambulance_standby_hospital[0]
    #         print(f'{self.step}> A{a.a_uid} assigned to H{hosp.h_uid}')
    #     self.ambulance_standby_hospital.pop(0)#受け渡し後のリストから削除
    #     if verb:
    #         print(f'Resources:{len(self.ambulance_standby_hospital)}')
    #         print('========')
    #     return key

    # def check_path(self,max_iter,a):
    #     max_iter += 1
    #     if a.a_mypath == None:
    #         if max_iter < 10:
    #             return True
    #         else:
    #             print(f'A{a.a_uid} has path={a.a_mypath} and max_iter={max_iter}')
    #             raise
    #     return False

    # def event_schedule(self,verb=False):
    #     'calculating next event'
    #     amb_events = [a.a_mytime for a in self.ambulances]
    #     amb_events = list(filter(None, amb_events))
    #     amb_events.sort()
    #     if len(amb_events)!=0:
    #         closest_event = amb_events[0]
    #         # if verb:
    #         #     print(f'{self.step}> Closest Event: {closest_event}')
    #         return closest_event
    #     return None

    # def go(self,sim_time,saveplot=False,width=10,height=10,dpi=300,video=False,fps=1,verb=False):
    #     'Main loop'
    #     if verb:
    #         print('Starting loop')
    #     for t in tqdm_notebook(range(sim_time)):
    #         self.step = t

    #         #assigning resources
    #         resources = len(self.ambulance_standby_injury)
    #         demand = len(self.injury_wait_list)
    #         if resources == 0 or demand == 0:
    #             if verb:
    #                 print(f'Resources:{resources}, Demand:{demand}')
    #         else:
    #             if resources >= demand:
    #                 for i in range(demand):
    #                     self.assign_injury(verb=verb)
    #             else:
    #                 for i in range(resources):
    #                     self.assign_injury(verb=verb)

    #         ##assign hospital to waiting ambulances
    #         if self.ambulance_standby_hospital:
    #             max_iter = 0
    #             while self.assign_hospital(max_iter,verb=verb):
    #                 pass

    #         next_event = self.event_schedule(verb=verb)
    #         # moving based on event time
    #         if self.step == next_event or self.step == 0:
    #             if verb:
    #                 print(f'=== {self.step} ===')
    #             for i, a in enumerate(self.ambulances):
    #                 if verb:
    #                     print(f'{self.step}> A{a.a_uid} state is {a.a_state}')
    #                 if a.a_state:
    #                     a.check_arrived(self, verb=verb)
    #             if saveplot:
    #                 self.plot(step=self.step+1,saveplot=saveplot,case=self.id,width=width,height=height,dpi=dpi,labels=self.labels,interactive=False,verb=verb)

    #     if video:
    #         print('Processing video...')
    #         self.video(case=self.id,name='anim',fps=fps,verb=verb)

    #     ##入院患者リストに入っているか確認
    #     print('Finished!')

    # # def plot_case_screen():

    # def plot(self,step=0,saveplot=False,case='kochi',width=32,height=16,dpi=300,
    # labels=False,interactive=False,verb=False):
    #     if interactive:
    #         mpld3.enable_notebook()
    #     else:
    #         mpld3.disable_notebook()
    #     if saveplot:
    #         mpld3.disable_notebook()
    #         plt.ioff()
    #     else:
    #         plt.ion()
    # #plot network
    #     fig,ax = self.Env.e_plot(width=width,height=height,verb=verb)
    #     fig.set_size_inches(width,height)
    # #plot tsunami in nodes
    #     if tsunami_file != None and self.tsunami == True:
    #         fig,ax = self.Env.e_plot_tsunami_in_network(width=width,height=height,verb=verb)
    # #plot tsunami as raster
    #     if tsunami_file != None and self.tsunami == True:
    #         self.Env.e_show_tsunami(ax=ax,verb=verb)
    # #plot injuries
    #     x = [i.i_current_coordinates[0] for i in self.injuries]
    #     y = [i.i_current_coordinates[1] for i in self.injuries]
    #     c = [i.i_triage for i in self.injuries]
    #     ax.scatter(x,y,c=c,cmap=cmap_triage,marker='.',s=100)
    #     if labels:
    #         for i,j in enumerate(self.injuries):
    #             ax.annotate(f'P{j.i_uid}', (x[i]+5e-6, y[i]+5e-6),color='white',
    #                 path_effects=[pe.withStroke(linewidth=1.5, foreground='k')])
    # #plot hospitals
    #     x = [h.h_home_coordinate[0] for h in self.hospitals]
    #     y = [h.h_home_coordinate[1] for h in self.hospitals]
    #     # names = [h.h_name.encode("utf-8") for h in self.hospitals]
    #     names = [h.h_uid for h in self.hospitals]
    #     ax.scatter(x,y,c='blue',marker='+',s=300)
    #     if labels:
    #         for i, txt in enumerate(names):
    #             ax.annotate(f'H{txt}', (x[i]+5e-6, y[i]+5e-6),color='k',
    #                         path_effects=[pe.withStroke(linewidth=2, foreground="yellow")])
    # #plot ambulances #救急車だけプロットのやり方を変える#pathリストを更新したら，座標も更新
    #     x = [a.a_current_coordinate[0] for a in self.ambulances]
    #     y = [a.a_current_coordinate[1] for a in self.ambulances]
    #     names = [a.a_uid for a in self.ambulances]
    #     ax.scatter(x,y,c='cyan',edgecolors='k',marker='*',s=300)
    #     if labels:
    #         for i, txt in enumerate(names):
    #             ax.annotate(f'A{txt}', (x[i], y[i]),color='k',
    #                 path_effects=[pe.withStroke(linewidth=2, foreground="cyan")])
    #     xmax = ax.get_xlim()[1]
    #     ymin = ax.get_ylim()[0]
    # #plot clock
    #     self.plot_clock(step,ax,xmax,ymin)
    #     if saveplot:
    #         plt.savefig(f'./outputs/img/fig{step:010d}.png',dpi=dpi)
    #         plt.close('all')
    #     if verb:
    #         print(f'F{step:010d} plotted')
    #     ax.grid(False)

    # def plot_clock(self,step,ax,x,y):
    #     time = str(datetime.timedelta(seconds=step))
    #     # ax.text(x-0.0025,y,time,c='k',bbox=dict(boxstyle="square",fc="white",ec='k'))
    #     legend = self.plot_legend(time)
    #     plt.subplots_adjust(right=0.8)
    #     ax.text(0.93,0.15,legend,c='k',ha="right",va="bottom",
    #             transform=plt.gcf().transFigure,backgroundcolor='white',
    #             bbox=dict(boxstyle="round,pad=1",fc="white",ec='k',pad=0.5))

    # def plot_legend(self,time):
    #     remain = len(self.injury_wait_list)
    #     in_transport = len(self.injury_in_transport)
    #     transported = len(self.injury_transported)
    #     txt = f'''
    #     Case: {self.id.capitalize()}\n
    #     A:{len(self.ambulances)}, H:{len(self.hospitals)}, P:{len(self.injuries)}\n
    #     ==================\n
    #     Time: {time}\n
    #     Waiting: {remain}\n
    #     In transportation: {in_transport}\n
    #     Transported: {transported}
    #     '''
    #     txt = textwrap.dedent(txt)
    #     return txt

    # def video(self,case='case',name='anim',fps=1,verb=False):
    #     image_folder=f'./outputs/img'
    #     video_name = f'./outputs/videos/{case}_{name}.avi'
    #     images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]
    #     frame = cv2.imread(os.path.join(image_folder, images[0]))
    #     height, width, layers = frame.shape
    #     video = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc('M','J','P','G'),
    #                             fps, (width,height))
    #     for image in images:
    #         if verb:
    #             print(image)
    #         video.write(cv2.imread(os.path.join(image_folder, image)))
    #     cv2.destroyAllWindows()
    #     video.release()
