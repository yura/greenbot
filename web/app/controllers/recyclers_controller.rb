class RecyclersController < ApplicationController
  before_action :set_recycler, only: %i[ show edit update destroy ]

  # GET /recyclers or /recyclers.json
  def index
    @q = Recycler.ransack(params[:q])
    @recyclers = @q.result(distinct: true).includes(:categories)

    lon = @recyclers.map(&:lon).compact
    lat = @recyclers.map(&:lat).compact

    @center = { lon: lon.reduce(:+) / lon.size, lat: lat.reduce(:+) / lat.size }
  end

  # GET /recyclers/1 or /recyclers/1.json
  def show
  end

  # GET /recyclers/new
  def new
    @recycler = Recycler.new
  end

  # GET /recyclers/1/edit
  def edit
  end

  # POST /recyclers or /recyclers.json
  def create
    @recycler = Recycler.new(recycler_params)

    respond_to do |format|
      if @recycler.save
        format.html { redirect_to recycler_url(@recycler), notice: "Recycler was successfully created." }
        format.json { render :show, status: :created, location: @recycler }
      else
        format.html { render :new, status: :unprocessable_entity }
        format.json { render json: @recycler.errors, status: :unprocessable_entity }
      end
    end
  end

  # PATCH/PUT /recyclers/1 or /recyclers/1.json
  def update
    respond_to do |format|
      if @recycler.update(recycler_params)
        format.html { redirect_to recycler_url(@recycler), notice: "Recycler was successfully updated." }
        format.json { render :show, status: :ok, location: @recycler }
      else
        format.html { render :edit, status: :unprocessable_entity }
        format.json { render json: @recycler.errors, status: :unprocessable_entity }
      end
    end
  end

  # DELETE /recyclers/1 or /recyclers/1.json
  def destroy
    @recycler.destroy

    respond_to do |format|
      format.html { redirect_to recyclers_url, notice: "Recycler was successfully destroyed." }
      format.json { head :no_content }
    end
  end

  private
    # Use callbacks to share common setup or constraints between actions.
    def set_recycler
      @recycler = Recycler.find(params[:id])
    end

    # Only allow a list of trusted parameters through.
    def recycler_params
      params.require(:recycler).permit(:name, :description, :phone, :email, :city_id, :address, :url, :lon, :lat, category_ids: [])
    end
end
