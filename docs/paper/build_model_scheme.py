"""Figure 1 model schematic (placeholder, replaceable)."""
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
NAVY="#21295C"; DEEP="#065A82"; TEAL="#1C7293"; GOLD="#E8A33D"; INK="#1A2230"
fig=Figure(figsize=(12,4.2),facecolor="white"); ax=fig.add_axes([0,0,1,1]); ax.axis("off"); ax.set_xlim(0,12); ax.set_ylim(0,4.2)
B=[(0.9,"NBED\npattern",DEEP),(3.0,"$\theta$-roll aug.\n(two rotated views)",TEAL),(5.4,"Truncated ResNet\nencoder (CNN, not ViT)",DEEP),(8.05,"Projector + prototypes\nDINO self-distillation\n(teacher/student, EMA)",DEEP),(10.7,"Emergent\nclass map\n($K$ not preset)",GOLD)]
w,h,y=1.7,1.2,2.5; cx=[]
for x,t,c in B:
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=0.06,rounding_size=0.12",fc=c,ec="none")); cx.append(x)
    ax.text(x,y,t,ha="center",va="center",color="white",fontsize=10.5,fontweight="bold")
for a,b in zip(cx[:-1],cx[1:]): ax.add_patch(FancyArrowPatch((a+w/2,y),(b-w/2,y),arrowstyle="-|>",mutation_scale=18,lw=2,color=INK))
ax.text(6.7,0.95,"Training losses:  radial-gated supervised-contrastive  +  1-D radial clustering loss",ha="center",va="center",fontsize=11,color=NAVY,fontweight="bold")
ax.text(6.7,0.45,"(optional: a few pairwise same/different hints  ->  semi-supervised steering)",ha="center",va="center",fontsize=9.5,color="#555",style="italic")
ax.annotate("",xy=(6.7,1.35),xytext=(6.7,1.85),arrowprops=dict(arrowstyle="-",lw=1.2,color="#999"))
ax.text(6.0,3.55,"Diffraction-tailored, label-free, no preset $K$",ha="center",fontsize=12.5,color=NAVY,fontweight="bold")
FigureCanvasAgg(fig); fig.savefig("docs/explainer/figs/model_scheme.png",dpi=200,facecolor="white",bbox_inches="tight"); print("wrote model_scheme.png")
