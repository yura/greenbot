class Category < ApplicationRecord
  has_many :recycler_categories
  has_many :recyclers, through: :recycler_categories
end
