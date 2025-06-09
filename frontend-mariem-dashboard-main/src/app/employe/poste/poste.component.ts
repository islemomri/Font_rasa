import { Component, Input, ViewChild, ChangeDetectorRef } from '@angular/core';
import { Table, TableModule } from 'primeng/table';
import { PosteAvecDatesDTO } from '../model/PosteAvecDatesDTO';
import { EmoloyeService } from '../service/emoloye.service';
import { CommonModule } from '@angular/common';
import { ButtonModule } from 'primeng/button';
import { ToolbarModule } from 'primeng/toolbar';
import { InputTextModule } from 'primeng/inputtext';
import { CalendarModule } from 'primeng/calendar';
import { DialogModule } from 'primeng/dialog';
import { ConfirmDialogModule } from 'primeng/confirmdialog';
import { FileUploadModule } from 'primeng/fileupload';
import { InputIconModule } from 'primeng/inputicon';
import { FormsModule } from '@angular/forms';
import { BadgeModule } from 'primeng/badge'; // Import the BadgeModule
import { PosteService } from '../../poste/service/poste.service';
import { DirectionService } from '../../direction/service/direction.service';
import { SiteService } from '../../site/service/site.service';
import { DropdownModule } from 'primeng/dropdown';
import { Direction } from '../../direction/model/Direction';
import { EmployePoste } from '../model/EmployePoste';
import { MultiSelectModule } from 'primeng/multiselect';
import { ConfirmationService, MessageService } from 'primeng/api';
import { MessageModule } from 'primeng/message';
import { formatDate } from '@angular/common';
import { TagModule } from 'primeng/tag';

@Component({
  selector: 'app-poste',
  imports: [
    TableModule,
    ButtonModule,
    ToolbarModule,
    InputTextModule,
    CalendarModule,
    CommonModule,
    DialogModule,
    ConfirmDialogModule,
    FileUploadModule,
    InputIconModule,
    FormsModule,
    BadgeModule,
    TagModule,
    DropdownModule,
    MultiSelectModule,
    MessageModule,
  ],
  templateUrl: './poste.component.html',
  styleUrls: ['./poste.component.css'],
  providers: [MessageService,ConfirmationService],
})
export class PosteComponent {
  postes: any[] = [];
  postesemploye: any[] = [];
  directions: Direction[] = [];
  sites: any[] = [];
  selectedPostes: any[] = [];
  posteDialog: boolean = false;
  selectedPosteDetails: any = {};
  poste: any = {};
  selectedPoste: any;
  selectedDirection: any;
  selectedSite: any;
  selectedPosteId: number | null = null;
  selectedDirectionId: number | null = null;
  selectedSiteId: number | null = null;
  @Input() employeId!: number;
  errorMessage: string = '';
  @ViewChild('dt') dt: Table | undefined;
  cols: any[] = [
    { field: 'posteId', header: 'ID Poste' },
    { field: 'dateDebut', header: 'Date Début' },
    { field: 'dateFin', header: 'Date Fin' },
    { field: 'statut', header: 'Statut' },
  ];
  errorMessageUpdate: string = '';
  constructor(
     private cdRef: ChangeDetectorRef,
    private posteService: PosteService,
    private directionService: DirectionService,
    private employeservice: EmoloyeService,
    private cdr: ChangeDetectorRef,
  private confirmationService: ConfirmationService,
    private messageService: MessageService
  ) {}

  ngOnInit(): void {
    // Remplacer par l'ID réel de l'employé
    this.employeservice.getPostesByEmploye(this.employeId).subscribe((data) => {
      this.postesemploye = data;
      console.log(this.postes);
    });

    this.posteService.getAllPostesnonArchives().subscribe((data) => {
      this.postes = data;
      console.log(this.postes);
    });
    console.log('selectedPosteDetails:', this.selectedPosteDetails);
    console.log('directions:', this.directions);
  }

openupdate(poste: any) {
  console.log('Poste sélectionné:', poste);

  if (!poste.employePosteId) {
    console.error('Le poste sélectionné n\'a pas d\'ID d\'association');
    return;
  }

  this.employeservice.getPosteDetailsById(poste.employePosteId).subscribe({
    next: (data: any) => {
      console.log('Détails reçus:', data);
      
      this.selectedPosteDetails = {
        ...data,
        employePosteId: poste.employePosteId,
        posteId: poste.posteId,
        dateDebut: data.dateDebut ? new Date(data.dateDebut) : null,
        dateFin: data.dateFin ? new Date(data.dateFin) : null
      };

      // Charger le poste sélectionné
      this.selectedPoste = this.postes.find(p => p.id === poste.posteId);

      // Charger les directions
      this.posteService.getDirectionsByPosteId(poste.posteId).subscribe(directions => {
        this.directions = directions;
        this.selectedDirection = this.directions.find(d => 
          d.id === data.directionId || d.nom_direction === data.nomDirection
        );
        
        // Charger les sites si une direction est sélectionnée
        if (this.selectedDirection) {
          this.directionService.getSitesByDirection(this.selectedDirection.id)
            .subscribe(sites => {
              this.sites = sites;
              this.selectedSite = this.sites.find(s => 
                s.id === data.siteId || s.nom_site === data.nomSite
              );
              this.updateDialog = true;
            });
        } else {
          this.updateDialog = true;
        }
      });
    },
    error: (err) => {
      console.error('Erreur détails:', err);
    }
  });
}
trackById(index: number, item: any): any {
  return item.employePosteId || item.id;
}

deletePoste(poste: any) {
  const employePosteId = poste.employePosteId || poste.id;

  if (!employePosteId) {
    this.messageService.add({
      severity: 'error',
      summary: 'Erreur',
      detail: 'ID manquant pour la suppression',
      life: 5000
    });
    return;
  }

  this.confirmationService.confirm({
    message: 'Êtes-vous sûr de vouloir supprimer ce poste ?',
    header: 'Confirmation de suppression',
    icon: 'pi pi-exclamation-triangle',
    acceptLabel: 'Oui',
    rejectLabel: 'Non',
    acceptButtonStyleClass: 'p-button-danger',
    rejectButtonStyleClass: 'p-button-text',
    accept: () => {
      this.employeservice.supprimerPosteParId(employePosteId).subscribe({
        next: () => {
          this.postesemploye = this.postesemploye.filter(
            p => p.employePosteId !== employePosteId
          );
          this.messageService.add({
            severity: 'success',
            summary: 'Succès',
            detail: 'Poste supprimé avec succès',
            life: 3000
          });
        },
        error: (err) => {
          this.messageService.add({
            severity: 'error',
            summary: 'Erreur',
            detail: 'Échec de la suppression: ' + (err.error?.message || err.message),
            life: 5000
          });
        }
      });
    }
  });
}

  

  formatDate(date: Date): string {
    const year = date.getFullYear();
    const month = (date.getMonth() + 1).toString().padStart(2, '0');
    const day = date.getDate().toString().padStart(2, '0');
    return `${year}-${month}-${day}`;
  }

  updateDialog: boolean = false;

  hideDialogUpdate() {
    this.updateDialog = false;
    this.errorMessageUpdate = ''; // Réinitialisation du message d'erreur
  }
  openNew() {
    this.posteDialog = true;
  }

  hideDialog() {
    this.posteDialog = false;
  }
isDateOverlap(newDebut: Date, newFin: Date | null, excludePosteId?: number): boolean {
  if (!newDebut) return false;
  
  for (const poste of this.postesemploye) {
    // Exclure le poste en cours de modification
    if (excludePosteId && poste.employePosteId === excludePosteId) {
      continue;
    }

    const debut = new Date(poste.dateDebut);
    const fin = poste.dateFin ? new Date(poste.dateFin) : null;

    const isOverlap =
      (newFin === null && (!fin || newDebut <= fin)) ||
      (fin === null && debut <= newFin!) ||
      (newDebut <= fin! && newFin! >= debut);

    if (isOverlap) {
      return true;
    }
  }
  return false;
}
  savePoste() {
    if (
      !this.selectedPoste ||
      !this.selectedDirection ||
      !this.selectedSite ||
      !this.poste.dateDebut
    ) {
      console.error('Certains champs sont manquants');
      return;
    }

    if (this.poste.dateFin) {
      const today = new Date();
      const selectedDateFin = new Date(this.poste.dateFin);

      if (selectedDateFin > today) {
        this.errorMessage =
          "Ce poste est en cours. Vous ne pouvez pas spécifier une date de fin avant qu'il ne soit terminé.";
        return;
      }

      const selectedDateDebut = new Date(this.poste.dateDebut);
      if (selectedDateDebut >= selectedDateFin) {
        this.errorMessage = 'La date de début doit être avant la date de fin.';
        return;
      }
    }

    // Vérification de dateFin null
    if (!this.poste.dateFin) {
      const posteEnCours = this.postesemploye.find((poste) => !poste.dateFin);
      if (posteEnCours) {
        this.errorMessage = 'Cet employé est déjà à un poste actuel.';
        return;
      }
    }
    const newDebut = this.poste.dateDebut;
  const newFin = this.poste.dateFin;
    if (this.isDateOverlap(newDebut, newFin)) {
    this.errorMessage = "Les dates du nouveau poste se chevauchent avec un autre poste.";
    return;
  }

    this.errorMessage = ''; // Réinitialisation de l'erreur

    const dateFin = this.poste.dateFin
      ? this.formatDate(this.poste.dateFin)
      : '';

    this.employeservice
      .ajouterPosteAEmploye(
        this.employeId,
        this.selectedPoste.id,
        this.selectedDirection.id,
        this.selectedSite.id,
        this.formatDate(this.poste.dateDebut),
        dateFin
      )
      .subscribe({
        next: (response) => {
          console.log('Poste ajouté avec succès :', response);
          // Mettre à jour la liste des postes
          this.employeservice
            .getPostesByEmploye(this.employeId)
            .subscribe((data) => {
              this.postesemploye = data;
            });
          this.posteDialog = false;
        },
        error: (err) => {
          console.error("Erreur lors de l'ajout du poste :", err);
        },
      });
  }

  onPosteSelect(event: any) {
    this.poste = event.value;
    const posteId = event.value.id; // Si l'ID du poste est disponible
    console.log('Poste sélectionné ID:', posteId);

    this.posteService.getDirectionsByPosteId(posteId).subscribe(
      (data) => {
        console.log('Directions récupérées:', data);
        this.directions = data;
      },
      (error) => {
        console.error('Erreur lors de la récupération des directions:', error);
      }
    );
  }

  onDirectionSelect(event: any) {
    const directionId = this.selectedDirection.id;
    console.log('Direction sélectionnée :', directionId);
    this.directionService.getSitesByDirection(directionId).subscribe((data) => {
      console.log('Sites pour cette direction :', data);
      this.sites = data;

      // Vérifier si les sites sont vides
      if (this.sites.length === 0) {
        console.log('Aucun site trouvé pour cette direction.');
        // Afficher un message ou faire une autre action
      }
    });
  }

  getPosteStatus(poste: PosteAvecDatesDTO): string {
    const today = new Date();
    const dateFin = poste.dateFin ? new Date(poste.dateFin) : null;

    // Si dateFin est null, le poste est en cours
    if (dateFin === null) {
      return 'En cours';
    }

    // Si dateFin est avant aujourd'hui, c'est un poste historique
    if (dateFin < today) {
      return 'Historique';
    }

    // Si dateFin est après aujourd'hui, le poste est en cours
    return 'En cours';
  }

  getBadgeSeverity(
    poste: PosteAvecDatesDTO
  ): 'info' | 'success' | 'warn' | 'danger' {
    const status = this.getPosteStatus(poste);
    switch (status) {
      case 'En cours':
        return 'info';
      case 'Terminé':
        return 'success';
      case 'À venir':
        return 'warn';
      default:
        return 'danger';
    }
  }

  exportCSV(event: Event) {
    console.log('Exporting CSV', event);
  }
  editPoste(poste: any) {
    this.employeservice
      .getPosteDetails(this.employeId, poste.posteId)
      .subscribe({
        next: (response: any) => {
          console.log('📌 Détails du poste :', response);

          if (!response) {
            console.error('⚠️ Données invalides !');
            return;
          }

          // Remplir les informations du poste
          this.poste = {
            dateDebut: response.dateDebut ? new Date(response.dateDebut) : null,
            dateFin: response.dateFin ? new Date(response.dateFin) : null,
          };

          // Charger les directions du poste
          this.posteService
            .getDirectionsByPosteId(poste.posteId)
            .subscribe((directionsData) => {
              this.directions = directionsData;
              this.selectedDirection =
                this.directions.find(
                  (d) => d.nom_direction === response.nomDirection
                ) || null;

              // Charger les sites de la direction sélectionnée
              if (this.selectedDirection) {
                this.directionService
                  .getSitesByDirection(this.selectedDirection.id)
                  .subscribe((sitesData) => {
                    this.sites = sitesData;
                    this.selectedSite =
                      this.sites.find((s) => s.nom_site === response.nomSite) ||
                      null;
                  });
              }
            });
        },
        error: (err) => console.error('❌ Erreur API :', err),
      });
  }

  onInputChange(event: Event) {
    const inputElement = event.target as HTMLInputElement;
    if (this.dt) {
      this.dt.filterGlobal(inputElement.value, 'contains');
    }
  }

  onSiteSelect(event: any) {
    console.log('Site sélectionné :', event.value);
    this.selectedSite = event.value || {}; // Évite l'erreur undefined
  }
modifierPoste() {
  // Réinitialiser le message d'erreur
  this.errorMessageUpdate = '';

  // Vérification des champs obligatoires
  if (!this.selectedPoste || !this.selectedDirection || !this.selectedSite || !this.selectedPosteDetails.dateDebut) {
    this.errorMessageUpdate = 'Certains champs sont manquants';
    return;
  }

  // Vérification des dates
  const newDebut = new Date(this.selectedPosteDetails.dateDebut);
  const newFin = this.selectedPosteDetails.dateFin ? new Date(this.selectedPosteDetails.dateFin) : null;
  const today = new Date();

  // Vérifier si dateFin est dans le futur
  if (newFin && newFin > today) {
    this.errorMessageUpdate = "Ce poste est en cours. Vous ne pouvez pas spécifier une date de fin future.";
    return;
  }

  // Vérifier si dateDebut est après dateFin
  if (newFin && newDebut >= newFin) {
    this.errorMessageUpdate = 'La date de début doit être avant la date de fin.';
    return;
  }

  // Vérifier les chevauchements de dates (en excluant le poste en cours de modification)
  if (this.isDateOverlap(newDebut, newFin, this.selectedPosteDetails.employePosteId)) {
    this.errorMessageUpdate = "Les dates du nouveau poste se chevauchent avec un autre poste.";
    return;
  }

  // Vérifier s'il y a déjà un poste en cours (sauf celui en cours de modification)
  if (!newFin) {
    const posteEnCours = this.postesemploye.find(
      poste => !poste.dateFin && poste.employePosteId !== this.selectedPosteDetails.employePosteId
    );
    if (posteEnCours) {
      this.errorMessageUpdate = 'Cet employé est déjà à un poste actuel.';
      return;
    }
  }

  // Si toutes les validations passent, procéder à la modification
  try {
    const dateDebut = this.formatDateToBackend(this.selectedPosteDetails.dateDebut);
    const dateFin = this.selectedPosteDetails.dateFin 
      ? this.formatDateToBackend(this.selectedPosteDetails.dateFin)
      : null;

    this.employeservice.modifierPosteComplet(
      this.selectedPosteDetails.employePosteId,
      this.selectedPoste?.id,
      this.selectedDirection.id,
      this.selectedSite.id,
      dateDebut,
      dateFin
    ).subscribe({
      next: (response) => {
          this.employeservice.getPostesByEmploye(this.employeId).subscribe((data) => {
            this.postesemploye = data;
          });

        console.log('Modification réussie:', response);
         this.employeservice
            .getPostesByEmploye(this.employeId)
            .subscribe((data) => {
              this.postesemploye = data;
            });
        this.messageService.add({
          severity: 'success',
          summary: 'Succès',
          detail: 'Poste mis à jour'
        });
        
        // Rafraîchir la liste des postes
        this.employeservice.getPostesByEmploye(this.employeId).subscribe((data) => {
          this.postesemploye = data;
        });
        
        this.updateDialog = false;
      },
      error: (err) => {
        console.error('Erreur modification:', err);
        this.errorMessageUpdate = err.error?.message || 'Erreur serveur';
      }
    });
  } catch (e) {
    console.error('Erreur conversion dates:', e);
    this.errorMessageUpdate = 'Format de date invalide';
  }
}

private formatDateToBackend(date: any): string {
  if (!date) {
      throw new Error('Date non fournie');
  }
  
  // Si c'est une string au format dd/mm/yyyy
  if (typeof date === 'string' && date.includes('/')) {
      const [day, month, year] = date.split('/');
      return `${year}-${month.padStart(2, '0')}-${day.padStart(2, '0')}`;
  }
  
  // Si c'est un objet Date
  if (date instanceof Date) {
      return date.toISOString().split('T')[0];
  }
  
  throw new Error(`Format de date non reconnu: ${date}`);
}
  areAllFieldsFilled(): boolean {
    if (
      !this.selectedPosteDetails?.dateDebut ||
      !this.selectedPosteDetails?.dateFin ||
      !this.selectedDirection ||
      !this.selectedSite ||
      !this.poste
    ) {
      return false;
    }

    // Vérifiez que dateDebut est avant dateFin
    const dateDebut = new Date(this.selectedPosteDetails.dateDebut);
    const dateFin = new Date(this.selectedPosteDetails.dateFin);
    return dateDebut < dateFin;
  }

  isUpdateFormValid(): boolean {
    return !!(
        this.selectedPosteDetails?.dateDebut &&
        this.selectedDirection &&
        this.selectedSite &&
        this.selectedPoste
    );
}

  getStatusIcon(poste: any): string {
    const status = this.getPosteStatus(poste);
    switch (status) {
      case 'En cours':
        return 'pi pi-spinner';
      case 'Terminé':
        return 'pi pi-check-circle';
      case 'Historique':
        return 'pi pi-history';
      default:
        return 'pi pi-question-circle';
    }
  }

  isFormValid(): boolean {
    return !!(
      this.selectedPoste && 
      this.selectedDirection && 
      this.selectedSite && 
      this.poste.dateDebut
    );
  }
  
  
}