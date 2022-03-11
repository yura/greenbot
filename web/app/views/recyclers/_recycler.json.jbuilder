json.extract! recycler, :id, :name, :description, :phone, :email, :city_id, :address, :url, :created_at, :updated_at
json.url recycler_url(recycler, format: :json)
